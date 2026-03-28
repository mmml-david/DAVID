"""Evaluate Qwen3-VL and DAVID+VAE on PerceptionTest videos with a VQA-style prompt.

This script supports three modes:
  - qwen: baseline Qwen3-VL generation from raw video.
  - david: DAVID VAE reconstruction injected before generation.
  - both: run both methods for side-by-side comparison.

Example:
  python evaluate_vqa.py \
    --method both \
    --split validation \
    --max_samples 100 \
    --prompt "What is the video caption?" \
    --vae_checkpoint ./checkpoints/step_0050000.pt
"""

from __future__ import annotations

import argparse
import io
import json
import os
import types
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from datasets import Video, load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from david.utils import pad_sequence_to_max
from david.vae import DAVIDConfig, DAVIDVAE


@dataclass
class EvalSample:
    row_index: int
    video_name: str
    label: Any
    video: Any


@dataclass
class VQASample:
    """A single VQA question loaded from a JSON file."""
    sample_id: str
    video_name: str
    video_path: str
    question: str
    options: list[str]
    answer_index: int
    split_name: str
    prompt: str  # formatted question + options ready for the model


def match_mcq_response(response: str, answer_index: int, options: list[str]) -> bool:
    """Check if a model response matches the correct answer.

    Handles:
      - Response starts with the correct letter: "C" or "C. moving"
      - Response contains the correct letter standalone: "The answer is C"
      - Response contains the correct option text: "The answer is: yes, it's a static."
    """
    import re

    if not response or answer_index < 0 or answer_index >= len(options):
        return False

    response_clean = response.strip()
    response_upper = response_clean.upper()
    option_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    correct_letter = option_labels[answer_index]
    correct_text = options[answer_index].strip().lower()

    # 1. Response starts with the correct letter (e.g. "C", "C.", "C. moving")
    if response_upper.startswith(correct_letter):
        # Make sure it's the letter alone, not a word starting with that letter
        if len(response_clean) == 1 or not response_clean[1].isalpha():
            return True

    # 2. Response contains the correct letter as a standalone token
    #    e.g. "The answer is C" or "Answer: B"
    if re.search(rf'\b{correct_letter}\b', response_upper):
        # Make sure no OTHER option letter also appears standalone (ambiguous)
        other_letters = [option_labels[i] for i in range(len(options)) if i != answer_index]
        if not any(re.search(rf'\b{ltr}\b', response_upper) for ltr in other_letters):
            return True

    # 3. Response contains the correct option text (fuzzy text match)
    #    e.g. "The answer is: yes, it's a static." matching option "static or shaking"
    response_lower = response_clean.lower()
    if correct_text and correct_text in response_lower:
        return True

    # 4. Check if the correct option text is the best match among all options
    #    (the response contains the correct option text but not other options)
    matches = [i for i, opt in enumerate(options) if opt.strip().lower() in response_lower]
    if len(matches) == 1 and matches[0] == answer_index:
        return True

    return False


def build_mcq_prompt(question: str, options: list[str]) -> str:
    """Format a multiple-choice question into a prompt string."""
    option_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [question]
    for i, opt in enumerate(options):
        label = option_labels[i] if i < len(option_labels) else str(i)
        lines.append(f"{label}. {opt}")
    lines.append("Answer with the letter of the correct option.")
    return "\n".join(lines)


def resolve_video_url_to_local(video_url: str, cache_root: str) -> str | None:
    """Resolve a HuggingFace video URL to a local cache path.

    Looks up the file under the HF hub cache directory structure:
      <cache_root>/datasets--<owner>--<repo>/snapshots/<hash>/<path_in_repo>
    """
    import re

    m = re.match(
        r"https://huggingface\.co/datasets/([^/]+)/([^/]+)/resolve/[^/]+/(.+)",
        video_url,
    )
    if m is None:
        return None
    owner, repo, path_in_repo = m.group(1), m.group(2), m.group(3)

    dataset_dir = os.path.join(cache_root, f"datasets--{owner}--{repo}")
    snapshots_dir = os.path.join(dataset_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    # Check each snapshot revision for the file
    for revision in os.listdir(snapshots_dir):
        candidate = os.path.join(snapshots_dir, revision, path_in_repo)
        if os.path.exists(candidate):
            return candidate
    return None


def load_vqa_samples_from_json(
    json_path: str,
    max_samples: int,
    cache_root: str,
    video_root: str | None = None,
) -> list[VQASample]:
    """Load valid-set VQA samples from a JSON file and resolve video paths."""
    with open(json_path) as f:
        data = json.load(f)

    # Filter to valid set only (video_url contains /valid/)
    valid_entries = [e for e in data if "/valid/" in e.get("video_url", "")]

    samples: list[VQASample] = []
    skipped = 0
    for entry in valid_entries:
        video_url = entry["video_url"]
        # Extract video name from URL
        video_name = os.path.splitext(os.path.basename(video_url))[0]

        # Resolve to local path: try video_root first, then HF cache
        local_path = None
        if video_root is not None:
            # Extract relative path after dataset name (e.g. "PerceptionTest/valid/video_X.mp4")
            # from URL like ".../resolve/main/PerceptionTest/valid/video_X.mp4"
            import re
            m = re.search(r"/resolve/[^/]+/(.+)$", video_url)
            if m:
                candidate = os.path.join(video_root, m.group(1))
                if os.path.exists(candidate):
                    local_path = candidate
        if local_path is None:
            local_path = resolve_video_url_to_local(video_url, cache_root)
        if local_path is None or not os.path.exists(local_path):
            skipped += 1
            continue

        question = entry.get("question", "")
        options = entry.get("option", [])
        answer_index = entry.get("answer", -1)
        prompt = build_mcq_prompt(question, options)

        samples.append(VQASample(
            sample_id=entry.get("sample_id", ""),
            video_name=video_name,
            video_path=local_path,
            question=question,
            options=options,
            answer_index=answer_index,
            split_name=entry.get("split_name", ""),
            prompt=prompt,
        ))

        if 0 < max_samples <= len(samples):
            break

    if skipped:
        print(f"Warning: skipped {skipped} entries whose videos were not found locally")
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VQA-style prompting on PerceptionTest videos")

    # Evaluation mode
    parser.add_argument("--method", choices=["qwen", "david", "both"], default="both")

    # Model args
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', 'cpu', etc.")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--attn_implementation", default="sdpa", help="Set to empty string to disable")

    # Dataset args
    parser.add_argument("--dataset_name", default="chancharikm/QualityCheck")
    parser.add_argument("--subset", default="PerceptionTest")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dedupe_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument(
        "--questions_json",
        default=None,
        help="Path to a JSON file with per-sample VQA questions (e.g. perception_test.json). "
             "When provided, only valid-set entries are used and each sample's question+options "
             "become the prompt. Overrides --prompt, --split, --dataset_name, etc.",
    )
    parser.add_argument(
        "--hf_cache_root",
        default="/DATA/huggingface/hub",
        help="Root directory of the HuggingFace hub cache where dataset blobs/snapshots live.",
    )
    parser.add_argument(
        "--video_root",
        default=None,
        help="Local directory containing dataset videos (e.g. /DATA/dataset/PerceptionTest). "
             "Tried before HF cache resolution.",
    )

    # Video / prompt args
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--prompt", default="What is the video caption?")

    # Generation args
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Visual token truncation (for baseline Qwen ablation)
    parser.add_argument("--visual_prefix_ratio", type=float, default=None,
                        help="Keep only the first X%% of visual tokens before LLM (e.g. 0.25 = 25%%). "
                             "Applied to both qwen and david methods.")

    # DAVID args
    parser.add_argument("--vae_device", default="auto", help="'auto' (same as model), 'cpu', or 'cuda:X'")
    parser.add_argument("--vae_config", default="configs/train_config.yaml")
    parser.add_argument("--vae_checkpoint", default=None)
    parser.add_argument("--vae_use_mu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae_prefix_tokens", type=int, default=None)
    parser.add_argument("--vae_prefix_ratio", type=float, default=None)
    parser.add_argument(
        "--deepstack_strategy",
        choices=["keep", "zero", "match_recon"],
        default="keep",
        help="How to handle DeepStack visual features in DAVID mode",
    )

    # Output args
    parser.add_argument("--output_dir", default="./eval_outputs")

    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def choose_methods(method_arg: str) -> list[str]:
    if method_arg == "both":
        return ["qwen", "david"]
    return [method_arg]


def video_name_from_entry(video_entry: Any, fallback_idx: int) -> str:
    if isinstance(video_entry, dict):
        path = video_entry.get("path")
        if path:
            return os.path.splitext(os.path.basename(path))[0]
    return f"sample_{fallback_idx:08d}"


def iter_eval_samples(
    dataset_name: str,
    subset: str,
    split: str,
    streaming: bool,
    dedupe_videos: bool,
    max_samples: int,
) -> Iterable[EvalSample]:
    ds = load_dataset(dataset_name, data_dir=subset, split=split, streaming=streaming)
    if "video" in ds.column_names:
        ds = ds.cast_column("video", Video(decode=False))

    seen: set[str] = set()
    yielded = 0

    for row_idx, sample in enumerate(ds):
        video_entry = sample.get("video")
        if video_entry is None:
            continue

        video_name = video_name_from_entry(video_entry, row_idx)
        if dedupe_videos and video_name in seen:
            continue
        seen.add(video_name)

        yield EvalSample(
            row_index=row_idx,
            video_name=video_name,
            label=sample.get("label"),
            video=video_entry,
        )
        yielded += 1
        if max_samples > 0 and yielded >= max_samples:
            return


def sample_frame_indices_at_fps(
    total_frames: int,
    video_fps: float,
    sample_fps: float,
    max_frames: int,
) -> list[int]:
    if total_frames <= 0:
        return []

    safe_video_fps = video_fps if video_fps and video_fps > 0 else sample_fps
    stride = max(safe_video_fps / sample_fps, 1.0)
    n = min(max(int(total_frames / stride), 1), max_frames)

    indices = [min(total_frames - 1, int(i * stride)) for i in range(n)]
    # Remove duplicates while preserving order
    return list(dict.fromkeys(indices))


def _resolve_hf_to_local(hf_url: str) -> str | None:
    """Resolve an hf://datasets/... URL to a local cache path if available."""
    import re

    from huggingface_hub import try_to_load_from_cache

    m = re.match(r"hf://datasets/([^@]+)@([^/]+)/(.+)", hf_url)
    if m is None:
        return None
    repo_id, revision, path_in_repo = m.group(1), m.group(2), m.group(3)
    result = try_to_load_from_cache(
        repo_id=repo_id,
        filename=path_in_repo,
        revision=revision,
        repo_type="dataset",
    )
    if isinstance(result, str):
        return result
    return None


def decode_video_frames(video_source: Any, sample_fps: float, max_frames: int) -> tuple[list[Image.Image], float]:
    import decord

    if isinstance(video_source, dict):
        path = video_source.get("path")
        blob = video_source.get("bytes")

        if path and os.path.exists(path):
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
        elif blob is not None:
            vr = decord.VideoReader(io.BytesIO(blob), ctx=decord.cpu(0))
        elif path is not None:
            if path.startswith("hf://"):
                local = _resolve_hf_to_local(path)
                if local is not None:
                    vr = decord.VideoReader(local, ctx=decord.cpu(0))
                else:
                    import fsspec

                    with fsspec.open(path, "rb") as f:
                        vr = decord.VideoReader(io.BytesIO(f.read()), ctx=decord.cpu(0))
            else:
                vr = decord.VideoReader(path, ctx=decord.cpu(0))
        else:
            raise ValueError("Video dict has neither usable 'path' nor 'bytes'")
    elif isinstance(video_source, bytes):
        vr = decord.VideoReader(io.BytesIO(video_source), ctx=decord.cpu(0))
    else:
        vr = decord.VideoReader(video_source, ctx=decord.cpu(0))

    video_fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    indices = sample_frame_indices_at_fps(total_frames, video_fps, sample_fps, max_frames)
    if not indices:
        raise ValueError("No frames sampled from video")

    frames = vr.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(frame) for frame in frames]
    return pil_frames, video_fps


def build_qwen_inputs(
    processor: AutoProcessor,
    frame_list: list[Image.Image],
    prompt: str,
    sample_fps: float,
    device: torch.device,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_list,
                    "fps": sample_fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    # For PIL frame lists, pass explicit metadata to preserve timestamps
    video_kwargs.pop("fps", None)
    inputs = processor(
        text=[text],
        videos=video_inputs,
        video_metadata=[
            {
                "fps": sample_fps,
                "total_num_frames": len(frame_list),
                "frames_indices": list(range(len(frame_list))),
            }
        ],
        **video_kwargs,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)

    # Qwen3-VL inserts per-frame timestamps, creating one video token group per
    # temporal frame.  get_rope_index expects one video_grid_thw row per group,
    # but the processor emits a single [T, H, W] row.  Expand it to T×[1, H, W].
    if "video_grid_thw" in inputs:
        grid = inputs["video_grid_thw"]
        expanded_rows = []
        for row in grid:
            t, h, w = row.tolist()
            t = int(t)
            expanded_rows.extend([[1, int(h), int(w)]] * t)
        inputs["video_grid_thw"] = torch.tensor(expanded_rows, dtype=grid.dtype, device=grid.device)

    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return dict(inputs)


def decode_generation(
    processor: AutoProcessor,
    generated_ids: torch.LongTensor,
    input_ids: torch.LongTensor,
) -> str:
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text[0].strip() if text else ""


def load_vae(vae_config_path: str, vae_checkpoint_path: str, device: torch.device) -> DAVIDVAE:
    with open(vae_config_path) as f:
        cfg = yaml.safe_load(f)

    david_cfg = DAVIDConfig.from_dict(dict(cfg.get("model", {})))
    vae = DAVIDVAE(david_cfg).to(device)

    state = torch.load(vae_checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    vae.load_state_dict(state_dict, strict=True)
    vae.eval()
    return vae


class VisualPrefixTruncator:
    """Zero-out visual tokens beyond the first `ratio` fraction (keeps token count unchanged)."""

    def __init__(self, qwen_model, ratio: float | None):
        self.ratio = ratio
        self._target = qwen_model.model
        self._orig = None

    @staticmethod
    def _zero_suffix(embeds, ratio):
        """Keep the first `ratio` fraction of tokens, zero out the rest."""
        out = []
        for emb in embeds:
            keep = max(1, int(emb.shape[0] * ratio))
            zeroed = emb.clone()
            zeroed[keep:] = 0
            out.append(zeroed)
        return out

    def __enter__(self):
        if self.ratio is None or self.ratio >= 1.0:
            return self
        orig = self._target.get_video_features

        def truncated_get_video_features(this, pixel_values_videos, video_grid_thw, **kwargs):
            result = orig(pixel_values_videos, video_grid_thw, **kwargs)
            if isinstance(result, tuple):
                video_embeds, deepstack = result
                return VisualPrefixTruncator._zero_suffix(video_embeds, self.ratio), deepstack
            else:
                result.pooler_output = VisualPrefixTruncator._zero_suffix(
                    result.pooler_output, self.ratio)
                return result

        self._orig = orig
        self._target.get_video_features = types.MethodType(truncated_get_video_features, self._target)
        return self

    def __exit__(self, *args):
        if self._orig is not None:
            self._target.get_video_features = self._orig
            self._orig = None


class DavidVideoFeatureAdapter:
    """Temporarily replaces Qwen3-VL video features with DAVID reconstructions."""

    def __init__(
        self,
        qwen_model: Qwen3VLForConditionalGeneration,
        vae: DAVIDVAE,
        use_mu: bool,
        prefix_tokens: int | None,
        prefix_ratio: float | None,
        deepstack_strategy: str,
    ):
        self.qwen_model = qwen_model
        self.vae = vae
        self.vae_device = next(vae.parameters()).device
        self.use_mu = use_mu
        self.prefix_tokens = prefix_tokens
        self.prefix_ratio = prefix_ratio
        self.deepstack_strategy = deepstack_strategy
        self._target = qwen_model.model
        self._orig_get_video_features = None

    def _prefix_len(self, n_tokens: int) -> int:
        if self.prefix_tokens is not None:
            return max(1, min(self.prefix_tokens, n_tokens))
        if self.prefix_ratio is not None:
            return max(1, min(int(round(n_tokens * self.prefix_ratio)), n_tokens))
        return n_tokens

    def _reconstruct_video_features(self, video_embeds: list[torch.Tensor]) -> list[torch.Tensor]:
        lengths = [emb.shape[0] for emb in video_embeds]

        features_list = [emb.float().to(self.vae_device) for emb in video_embeds]
        features, mask = pad_sequence_to_max(features_list)
        features = features.to(self.vae_device)
        mask = mask.to(self.vae_device)

        with torch.no_grad():
            z, mu, _ = self.vae.encode(features, mask)
            latent = mu if self.use_mu else z

            recon_list: list[torch.Tensor] = []
            for i, n_tokens in enumerate(lengths):
                m = self._prefix_len(n_tokens)
                recon = self.vae.decode(latent[i:i + 1, :m, :], n=n_tokens)[0]
                recon = recon.to(device=video_embeds[i].device, dtype=video_embeds[i].dtype)
                recon_list.append(recon)

        return recon_list

    def _adapt_deepstack(
        self,
        deepstack_video_embeds: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
        recon_video_embeds: list[torch.Tensor],
    ) -> list[torch.Tensor] | tuple[torch.Tensor, ...] | None:
        if deepstack_video_embeds is None:
            return None

        if self.deepstack_strategy == "keep":
            return deepstack_video_embeds

        if self.deepstack_strategy == "zero":
            out = [torch.zeros_like(x) for x in deepstack_video_embeds]
            return tuple(out) if isinstance(deepstack_video_embeds, tuple) else out

        # match_recon
        recon_flat = torch.cat(recon_video_embeds, dim=0)
        out = []
        for x in deepstack_video_embeds:
            if x.shape[0] == recon_flat.shape[0] and x.shape[-1] == recon_flat.shape[-1]:
                out.append(recon_flat.to(device=x.device, dtype=x.dtype))
            else:
                out.append(x)
        return tuple(out) if isinstance(deepstack_video_embeds, tuple) else out

    def __enter__(self):
        self._orig_get_video_features = self._target.get_video_features

        def patched_get_video_features(this, pixel_values_videos, video_grid_thw, **kwargs):
            video_outputs = self._orig_get_video_features(pixel_values_videos, video_grid_thw, **kwargs)
            # Handle both tuple return (newer transformers) and named-attribute return (older)
            if isinstance(video_outputs, tuple):
                video_embeds, deepstack_video_embeds = video_outputs
                recon_video_embeds = self._reconstruct_video_features(video_embeds)
                adapted_deepstack = self._adapt_deepstack(deepstack_video_embeds, recon_video_embeds)
                return recon_video_embeds, adapted_deepstack
            else:
                video_embeds = video_outputs.pooler_output
                deepstack_video_embeds = video_outputs.deepstack_features
                recon_video_embeds = self._reconstruct_video_features(video_embeds)
                adapted_deepstack = self._adapt_deepstack(deepstack_video_embeds, recon_video_embeds)
                video_outputs.pooler_output = recon_video_embeds
                video_outputs.deepstack_features = adapted_deepstack
                return video_outputs

        self._target.get_video_features = types.MethodType(patched_get_video_features, self._target)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_get_video_features is not None:
            self._target.get_video_features = self._orig_get_video_features
        self._orig_get_video_features = None


def run_generation(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    inputs: dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    return decode_generation(processor, generated_ids, inputs["input_ids"])


def make_record(
    method: str,
    sample: EvalSample | VQASample,
    prompt: str,
    response: str | None,
    error: str | None,
    video_fps: float | None,
    n_frames: int | None,
    elapsed_sec: float,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "method": method,
        "video_name": sample.video_name if isinstance(sample, VQASample) else sample.video_name,
        "prompt": prompt,
        "response": response,
        "error": error,
        "video_fps": video_fps,
        "num_sampled_frames": n_frames,
        "elapsed_sec": elapsed_sec,
    }
    if isinstance(sample, VQASample):
        rec["sample_id"] = sample.sample_id
        rec["split_name"] = sample.split_name
        rec["question"] = sample.question
        rec["options"] = sample.options
        rec["answer_index"] = sample.answer_index
        option_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rec["answer_letter"] = option_labels[sample.answer_index] if 0 <= sample.answer_index < len(option_labels) else None
        rec["matched"] = match_mcq_response(response or "", sample.answer_index, sample.options)
    else:
        rec["row_index"] = sample.row_index
        rec["label"] = sample.label
    return rec


def main() -> None:
    args = parse_args()
    methods = choose_methods(args.method)

    if args.method in {"david", "both"} and not args.vae_checkpoint:
        print("Warning: no --vae_checkpoint provided, using randomly initialized VAE")

    if args.vae_prefix_tokens is not None and args.vae_prefix_ratio is not None:
        raise ValueError("Use only one of --vae_prefix_tokens or --vae_prefix_ratio.")

    if args.vae_prefix_ratio is not None and not (0.0 < args.vae_prefix_ratio <= 1.0):
        raise ValueError("--vae_prefix_ratio must be in (0, 1].")

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = resolve_dtype(args.torch_dtype, device_str)

    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"

    print(f"Loading Qwen3-VL: {args.model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    if args.device != "auto":
        model = model.to(device_str)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_name)
    model_device = next(model.parameters()).device
    print(f"Model ready on device: {model_device}")

    vae = None
    david_adapter = None
    if "david" in methods:
        vae_device = torch.device("cpu") if args.vae_device == "cpu" else torch.device(model_device)
        if args.vae_checkpoint:
            print(f"Loading DAVID VAE checkpoint: {args.vae_checkpoint} (device: {vae_device})")
            vae = load_vae(args.vae_config, args.vae_checkpoint, vae_device)
        else:
            print(f"Initializing random VAE on {vae_device}")
            with open(args.vae_config) as f:
                cfg = yaml.safe_load(f)
            vae = DAVIDVAE(DAVIDConfig.from_dict(dict(cfg.get("model", {})))).to(vae_device)
            vae.eval()
        david_adapter = DavidVideoFeatureAdapter(
            qwen_model=model,
            vae=vae,
            use_mu=args.vae_use_mu,
            prefix_tokens=args.vae_prefix_tokens,
            prefix_ratio=args.vae_prefix_ratio,
            deepstack_strategy=args.deepstack_strategy,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"perceptiontest_{args.split}_{args.method}_{timestamp}.jsonl"
    summary_path = out_dir / f"perceptiontest_{args.split}_{args.method}_{timestamp}_summary.json"

    records: list[dict[str, Any]] = []

    use_json = args.questions_json is not None

    if use_json:
        # ── JSON-based VQA evaluation (valid set) ──
        print(f"Loading VQA questions from: {args.questions_json}")
        vqa_samples = load_vqa_samples_from_json(args.questions_json, args.max_samples, args.hf_cache_root, args.video_root)
        print(f"Loaded {len(vqa_samples)} valid-set VQA samples")

        split_label = "valid"
        jsonl_path = out_dir / f"perceptiontest_{split_label}_{args.method}_{timestamp}.jsonl"
        summary_path = out_dir / f"perceptiontest_{split_label}_{args.method}_{timestamp}_summary.json"

        # Cache decoded frames per video to avoid re-decoding for multiple questions
        frame_cache: dict[str, tuple[list[Image.Image], float]] = {}

        progress = tqdm(vqa_samples, desc="Evaluating VQA")
        for vqa_sample in progress:
            prompt = vqa_sample.prompt

            try:
                if vqa_sample.video_path not in frame_cache:
                    frames, video_fps = decode_video_frames(
                        vqa_sample.video_path,
                        sample_fps=args.sample_fps,
                        max_frames=args.max_frames,
                    )
                    frame_cache[vqa_sample.video_path] = (frames, video_fps)
                frames, video_fps = frame_cache[vqa_sample.video_path]

                inputs = build_qwen_inputs(
                    processor=processor,
                    frame_list=frames,
                    prompt=prompt,
                    sample_fps=args.sample_fps,
                    device=model_device,
                )

                for method in methods:
                    t0 = datetime.now()
                    response = None
                    error = None

                    try:
                        david_ctx = nullcontext() if method == "qwen" else david_adapter
                        trunc_ctx = VisualPrefixTruncator(model, args.visual_prefix_ratio)
                        with trunc_ctx, david_ctx:
                            response = run_generation(
                                model=model,
                                processor=processor,
                                inputs=inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=args.do_sample,
                                temperature=args.temperature,
                                top_p=args.top_p,
                            )
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        error = f"{type(e).__name__}: {e}"

                    elapsed_sec = (datetime.now() - t0).total_seconds()
                    records.append(
                        make_record(
                            method=method,
                            sample=vqa_sample,
                            prompt=prompt,
                            response=response,
                            error=error,
                            video_fps=video_fps,
                            n_frames=len(frames),
                            elapsed_sec=elapsed_sec,
                        )
                    )

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                for method in methods:
                    records.append(
                        make_record(
                            method=method,
                            sample=vqa_sample,
                            prompt=prompt,
                            response=None,
                            error=err,
                            video_fps=None,
                            n_frames=None,
                            elapsed_sec=0.0,
                        )
                    )

    else:
        # ── Original HF-dataset-based evaluation ──
        jsonl_path = out_dir / f"perceptiontest_{args.split}_{args.method}_{timestamp}.jsonl"
        summary_path = out_dir / f"perceptiontest_{args.split}_{args.method}_{timestamp}_summary.json"

        print(
            f"Running evaluation: methods={methods}, split={args.split}, "
            f"streaming={args.streaming}, max_samples={args.max_samples}"
        )

        samples = iter_eval_samples(
            dataset_name=args.dataset_name,
            subset=args.subset,
            split=args.split,
            streaming=args.streaming,
            dedupe_videos=args.dedupe_videos,
            max_samples=args.max_samples,
        )

        progress = tqdm(samples, total=args.max_samples if args.max_samples > 0 else None, desc="Evaluating")

        for sample in progress:
            try:
                frames, video_fps = decode_video_frames(
                    sample.video,
                    sample_fps=args.sample_fps,
                    max_frames=args.max_frames,
                )

                inputs = build_qwen_inputs(
                    processor=processor,
                    frame_list=frames,
                    prompt=args.prompt,
                    sample_fps=args.sample_fps,
                    device=model_device,
                )

                for method in methods:
                    t0 = datetime.now()
                    response = None
                    error = None

                    try:
                        david_ctx = nullcontext() if method == "qwen" else david_adapter
                        trunc_ctx = VisualPrefixTruncator(model, args.visual_prefix_ratio)
                        with trunc_ctx, david_ctx:
                            response = run_generation(
                                model=model,
                                processor=processor,
                                inputs=inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=args.do_sample,
                                temperature=args.temperature,
                                top_p=args.top_p,
                            )
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        error = f"{type(e).__name__}: {e}"

                    elapsed_sec = (datetime.now() - t0).total_seconds()
                    records.append(
                        make_record(
                            method=method,
                            sample=sample,
                            prompt=args.prompt,
                            response=response,
                            error=error,
                            video_fps=video_fps,
                            n_frames=len(frames),
                            elapsed_sec=elapsed_sec,
                        )
                    )

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                for method in methods:
                    records.append(
                        make_record(
                            method=method,
                            sample=sample,
                            prompt=args.prompt,
                            response=None,
                            error=err,
                            video_fps=None,
                            n_frames=None,
                            elapsed_sec=0.0,
                        )
                    )

    # ── Write outputs ──
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "questions_json": args.questions_json,
        "dataset": {
            "dataset_name": args.dataset_name,
            "subset": args.subset,
            "split": "valid" if use_json else args.split,
            "streaming": args.streaming,
            "dedupe_videos": args.dedupe_videos,
            "max_samples": args.max_samples,
        },
        "prompt": "per-sample MCQ" if use_json else args.prompt,
        "visual_prefix_ratio": args.visual_prefix_ratio,
        "vae_prefix_ratio": args.vae_prefix_ratio,
        "methods": methods,
        "n_records": len(records),
        "n_errors": sum(1 for r in records if r["error"] is not None),
        "outputs": {
            "jsonl": str(jsonl_path),
        },
    }

    # Per-method quick stats (with accuracy for JSON mode)
    method_stats: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_rows = [r for r in records if r["method"] == method]
        ok_rows = [r for r in method_rows if r["error"] is None]
        avg_latency = sum(r["elapsed_sec"] for r in ok_rows) / len(ok_rows) if ok_rows else None
        stats: dict[str, Any] = {
            "n_records": len(method_rows),
            "n_ok": len(ok_rows),
            "n_errors": len(method_rows) - len(ok_rows),
            "avg_elapsed_sec": avg_latency,
        }
        if use_json and ok_rows:
            correct = sum(1 for r in ok_rows if r.get("matched"))
            stats["n_correct"] = correct
            stats["accuracy"] = correct / len(ok_rows)
        method_stats[method] = stats

    summary["method_stats"] = method_stats

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"Saved predictions: {jsonl_path}")
    print(f"Saved summary: {summary_path}")
    if use_json:
        for method, stats in method_stats.items():
            if "accuracy" in stats:
                print(f"  {method} accuracy: {stats['accuracy']:.2%} ({stats['n_correct']}/{stats['n_ok']})")


if __name__ == "__main__":
    main()
