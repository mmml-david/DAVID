"""One-time feature extraction script.

Downloads PerceptionTest videos from chancharikm/QualityCheck and extracts
Qwen3-VL visual features, saving them as .pt files for fast VAE training.

Output filenames use the video name from the dataset: {video_name}.pt
This naturally deduplicates (PerceptionTest has ~30k rows but only ~11.6k unique
videos — multiple QA pairs per video) and makes the cache self-documenting.
Files from different shards are still non-overlapping because each shard owns a
disjoint slice of the unique-video list.

Options are read from the extraction: section of the YAML config file, then
overridden by any CLI flags that are explicitly provided.

Single-machine usage (all settings from config):
    python extract_features.py --config configs/train_config.yaml

Override specific settings via CLI (takes precedence over config):
    python extract_features.py --config configs/train_config.yaml \
        --device cuda:1 --max_samples 100

Distributed usage — each machine sets its own --shard_id:
    # Machine 0 of 4:
    python extract_features.py --config configs/train_config.yaml \
        --num_shards 4 --shard_id 0 --cache_dir /shared/features_cache
    # Machine 1 of 4:
    python extract_features.py --config configs/train_config.yaml \
        --num_shards 4 --shard_id 1 --cache_dir /shared/features_cache

Each machine writes non-overlapping files named by their global index.
Re-running is safe — existing files are skipped (resume support).
"""

import argparse
import io
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-VL features from PerceptionTest videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Config file — all other flags are optional overrides
    parser.add_argument("--config", default="configs/train_config.yaml",
                        help="Path to YAML config file (uses extraction: section)")

    # All flags default to None so we can tell when the user explicitly set them.
    # Values from the config fill in anything left as None.
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--sample_fps", type=float, default=None,
                        help="Frames per second to sample from each video")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Safety cap on frames per video (in case of very long videos)")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (-1 = all)")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of machines processing in parallel")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Index of this machine (0-indexed, must be < num_shards)")
    return parser.parse_args()


def load_extraction_config(config_path: str) -> dict[str, Any]:
    """Load the extraction section from a YAML config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("extraction", {})


def resolve_args(args, config_path: str) -> argparse.Namespace:
    """Merge config file values with CLI args. CLI args take precedence."""
    cfg = load_extraction_config(config_path)

    # Fallback defaults (used if neither CLI nor config provides a value).
    # shard_id intentionally omitted — it must be set explicitly when num_shards > 1.
    defaults = {
        "cache_dir": "./features_cache",
        "split": "train",
        "subset": "PerceptionTest",
        "dataset_name": "chancharikm/QualityCheck",
        "sample_fps": 1.0,
        "max_frames": 64,
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "device": "cuda",
        "max_samples": -1,
        "num_shards": 1,
    }

    for key, default in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, cfg.get(key, default))

    # Resolve shard_id separately: no built-in default when num_shards > 1.
    if args.shard_id is None:
        args.shard_id = cfg.get("shard_id", None)

    if args.num_shards > 1 and args.shard_id is None:
        raise ValueError(
            f"--shard_id is required when --num_shards={args.num_shards} > 1. "
            "Set it via CLI (--shard_id N) or under extraction.shard_id in the config."
        )

    # When num_shards == 1, default shard_id to 0.
    if args.shard_id is None:
        args.shard_id = 0

    return args


def sample_frame_indices_at_fps(
    total_frames: int, video_fps: float, sample_fps: float, max_frames: int
) -> list[int]:
    """Return frame indices sampled at sample_fps, capped at max_frames.

    Args:
        total_frames: Total number of frames in the video.
        video_fps:    Native fps of the video (from decord).
        sample_fps:   Desired sampling rate (e.g. 1.0 for 1 fps).
        max_frames:   Hard cap to avoid OOM on unusually long videos.
    """
    stride = video_fps / sample_fps          # e.g. 30fps / 1fps = take every 30th frame
    n = min(int(total_frames / stride), max_frames)
    return [int(i * stride) for i in range(n)]


def _ensure_symlink(path: str) -> str:
    """Recreate a missing HF hub snapshot symlink from the already-downloaded local blob.

    The hub cache stores files as content-addressed blobs and exposes them via a
    symlink tree under snapshots/.  If the symlink is missing (e.g. interrupted
    snapshot_download) but the blob exists, hf_hub_download with local_files_only=True
    will recreate the symlink with no network I/O.

    Path format: <hub_cache>/datasets--<owner>--<repo>/snapshots/<revision>/<filename>
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    parts = path.split("/snapshots/")
    if len(parts) != 2:
        return path
    repo_dir = parts[0].rsplit("/", 1)
    hub_cache, repo_folder = repo_dir[0], repo_dir[1]  # e.g. datasets--owner--repo
    _, owner, repo = repo_folder.split("--", 2)
    revision, filename = parts[1].split("/", 1)
    try:
        return hf_hub_download(
            repo_id=f"{owner}/{repo}",
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=hub_cache,
            local_files_only=True,
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return path  # blob not present; caller will get a decord error


def extract_one(sample, backbone, processor, sample_fps: float, max_frames: int):
    """Extract Qwen3-VL features from a single dataset sample."""
    import decord
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    # Try common video field names
    video_source = None
    for field in ["video", "video_bytes", "mp4"]:
        if field in sample and sample[field] is not None:
            video_source = sample[field]
            break

    if video_source is None:
        return None, f"No video field. Keys: {list(sample.keys())}"

    try:
        if isinstance(video_source, dict):
            # datasets.Video(decode=False) returns {"path": ..., "bytes": ...}
            if video_source.get("path") is not None:
                path = video_source["path"]
                if not os.path.exists(path):
                    path = _ensure_symlink(path)
                vr = decord.VideoReader(path, ctx=decord.cpu(0))
            elif video_source.get("bytes") is not None:
                vr = decord.VideoReader(io.BytesIO(video_source["bytes"]), ctx=decord.cpu(0))
            else:
                return None, "Video dict has neither 'path' nor 'bytes'"
        elif isinstance(video_source, bytes):
            vr = decord.VideoReader(io.BytesIO(video_source), ctx=decord.cpu(0))
        else:
            vr = decord.VideoReader(video_source, ctx=decord.cpu(0))

        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        indices = sample_frame_indices_at_fps(total_frames, video_fps, sample_fps, max_frames)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    except Exception as e:
        return None, f"Video decode error: {e}"

    # qwen_vl_utils expects each frame to be path/url/PIL.Image in list/tuple form.
    frame_list = [Image.fromarray(frame) for frame in frames]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_list,
                    "fps": sample_fps,
                },
                {"type": "text", "text": "Describe the video."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    # process_vision_info cannot infer fps from PIL images; provide VideoMetadata
    # explicitly so Qwen3-VL builds correct per-frame timestamps (idx/fps seconds)
    # instead of defaulting to 24 fps. Pop fps from video_kwargs to avoid conflict.
    video_kwargs.pop("fps", None)

    inputs = processor(
        text=[text],
        videos=video_inputs,
        video_metadata=[{"fps": sample_fps, "total_num_frames": len(frame_list), "frames_indices": list(range(len(frame_list)))}],
        **video_kwargs,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values_videos"]
    grid_thw = inputs["video_grid_thw"]

    features, mask = backbone.extract_features(pixel_values, grid_thw)
    # features: [1, N, D], squeeze batch dim, save as float16.
    # Mask is not saved — each file is one complete video so all tokens are always valid.
    return {"features": features[0].cpu().half()}, None


def _video_name(video_entry: dict) -> str:
    """Extract the filename stem from a Video(decode=False) dict, e.g. 'video_1'."""
    return os.path.splitext(os.path.basename(video_entry["path"]))[0]


def main():
    args = parse_args()
    args = resolve_args(args, args.config)
    print(f"Config: {args.config}")
    print(f"  dataset={args.dataset_name}/{args.subset}  split={args.split}")
    print(f"  cache_dir={args.cache_dir}  model={args.model_name}  device={args.device}")
    print(f"  sample_fps={args.sample_fps}  max_frames={args.max_frames}  max_samples={args.max_samples}")
    print(f"  shard={args.shard_id}/{args.num_shards}")

    from datasets import Video, load_dataset
    from david.backbone import Qwen3VLBackbone

    # Setup output directory
    out_dir = Path(args.cache_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load backbone
    backbone = Qwen3VLBackbone(
        model_name=args.model_name,
        device=args.device,
        dtype=torch.bfloat16,
    )
    processor = backbone.get_processor()

    # Load dataset
    print(f"Loading dataset {args.dataset_name} / {args.subset} split={args.split}")
    ds = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
    )
    if "video" in ds.column_names:
        ds = ds.cast_column("video", Video(decode=False))

    # Deduplicate: PerceptionTest has ~30k rows but only ~11.6k unique videos
    # (multiple QA pairs per video). Build a mapping: video_name → first row index.
    # The dataset has a single "video" column; use the filename stem as the unique key.
    unique_videos: dict[str, int] = {}
    for row_idx, entry in enumerate(ds["video"]):
        name = _video_name(entry)
        if name not in unique_videos:
            unique_videos[name] = row_idx
    unique_video_list = list(unique_videos.items())  # [(video_name, row_idx), ...]
    print(f"Dataset rows: {len(ds)} → unique videos: {len(unique_video_list)}")

    assert 0 <= args.shard_id < args.num_shards, \
        f"shard_id={args.shard_id} must be in [0, num_shards={args.num_shards})"

    n_total = len(unique_video_list) if args.max_samples == -1 \
        else min(args.max_samples, len(unique_video_list))

    # Shard over unique videos; last shard absorbs any remainder.
    shard_size = n_total // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = n_total if args.shard_id == args.num_shards - 1 else start_idx + shard_size
    my_videos = unique_video_list[start_idx:end_idx]

    print(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(my_videos)} unique videos → {out_dir}"
    )

    n_saved = 0
    n_errors = 0

    for video_name, row_idx in tqdm(my_videos, desc=f"Shard {args.shard_id}"):
        out_path = out_dir / f"{video_name}.pt"
        if out_path.exists():
            n_saved += 1
            continue  # resume support

        sample = ds[row_idx]
        result, err = extract_one(sample, backbone, processor, args.sample_fps, args.max_frames)

        if err is not None:
            print(f"[WARN] {video_name}: {err}")
            n_errors += 1
            continue

        torch.save(result, out_path)
        n_saved += 1

    print(f"\nShard {args.shard_id} done. Saved: {n_saved}, Errors: {n_errors}")


if __name__ == "__main__":
    main()
