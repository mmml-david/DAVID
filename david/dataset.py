"""Dataset for PerceptionTest videos from chancharikm/QualityCheck.

Supports two modes:
  - cached: Load pre-extracted .pt feature files (fast, recommended for training).
  - online: Stream videos from HuggingFace, extract frames, run through backbone.
"""

import os
import io
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path


def _load_cached_feature(path: str) -> dict:
    """Load a cached feature file saved by extract_features.py."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    features = data["features"].float()  # cast from float16 to float32
    # Mask is not stored — each file is a single complete video so all tokens are valid.
    mask = torch.ones(features.shape[0], dtype=torch.bool)
    return {"features": features, "mask": mask}


class PerceptionTestVideoDataset(Dataset):
    """Dataset for pre-extracted Qwen3-VL features of PerceptionTest videos.

    In cached mode: expects feature_cache_dir/{split}/*.pt files.
    In online mode: streams videos from HuggingFace and extracts features on-the-fly.
    """

    def __init__(
        self,
        feature_cache_dir: str,
        split: str = "train",
        mode: str = "cached",
        # Online mode args (ignored in cached mode)
        hf_dataset_name: str = "chancharikm/QualityCheck",
        subset: str = "PerceptionTest",
        backbone=None,
        sample_fps: float = 1.0,
        max_frames: int = 64,
    ):
        self.mode = mode
        self.split = split
        self.sample_fps = sample_fps

        if mode == "cached":
            cache_path = Path(feature_cache_dir) / split
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Feature cache not found at {cache_path}. "
                    "Run extract_features.py first."
                )
            self.feature_files = sorted(cache_path.glob("*.pt"))
            if len(self.feature_files) == 0:
                raise ValueError(f"No .pt files found in {cache_path}")
            print(f"[Dataset] Cached mode: {len(self.feature_files)} samples from {cache_path}")

        elif mode == "online":
            from datasets import Video, load_dataset
            assert backbone is not None, "backbone required for online mode"
            self.backbone = backbone
            self.max_frames = max_frames
            self.processor = backbone.get_processor()

            print(f"[Dataset] Online mode: streaming {hf_dataset_name} / {subset}")
            ds = load_dataset(
                hf_dataset_name,
                data_dir=subset,
                split=split,
                streaming=False,
            )
            if "video" in ds.column_names:
                ds = ds.cast_column("video", Video(decode=False))
            self.dataset = ds
        else:
            raise ValueError(f"mode must be 'cached' or 'online', got {mode!r}")

    def __len__(self) -> int:
        if self.mode == "cached":
            return len(self.feature_files)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "cached":
            return _load_cached_feature(str(self.feature_files[idx]))
        else:
            return self._online_getitem(idx)

    def _online_getitem(self, idx: int) -> dict:
        """Extract features on-the-fly for a single video."""
        import decord
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        sample = self.dataset[idx]

        # Try common video field names
        video_source = None
        for field in ["video", "video_bytes", "mp4"]:
            if field in sample and sample[field] is not None:
                video_source = sample[field]
                break

        if video_source is None:
            raise ValueError(f"No video field found in sample {idx}. Keys: {list(sample.keys())}")

        # Decode video frames using decord
        if isinstance(video_source, dict):
            if video_source.get("path") is not None:
                vr = decord.VideoReader(video_source["path"], ctx=decord.cpu(0))
            elif video_source.get("bytes") is not None:
                vr = decord.VideoReader(io.BytesIO(video_source["bytes"]), ctx=decord.cpu(0))
            else:
                raise ValueError(
                    f"Video dict in sample {idx} has neither 'path' nor 'bytes': {video_source.keys()}"
                )
        elif isinstance(video_source, bytes):
            vr = decord.VideoReader(io.BytesIO(video_source), ctx=decord.cpu(0))
        else:
            vr = decord.VideoReader(video_source, ctx=decord.cpu(0))

        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        indices = _sample_frame_indices_at_fps(total_frames, video_fps, self.sample_fps, self.max_frames)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]

        # qwen_vl_utils expects each frame to be path/url/PIL.Image in list/tuple form.
        frame_list = [Image.fromarray(frame) for frame in frames]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_list,
                        "fps": self.sample_fps,
                    },
                    {"type": "text", "text": "Describe the video."},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
            video_kwargs["fps"] = video_kwargs["fps"][0]

        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            **video_kwargs,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values_videos"]  # [total_patches, C*t*h*w]
        grid_thw = inputs["video_grid_thw"]           # [1, 3]

        features, mask = self.backbone.extract_features(pixel_values, grid_thw)
        # features: [1, N, D], mask: [1, N] — squeeze batch dim
        return {"features": features[0].cpu(), "mask": mask[0].cpu()}

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Pad variable-length features to a uniform batch tensor.

        Args:
            batch: List of dicts with "features" [N_i, D] and "mask" [N_i].

        Returns:
            Dict with:
                features: [B, N_max, D]
                mask:     [B, N_max] bool
        """
        from .utils import pad_sequence_to_max

        feature_list = [item["features"] for item in batch]
        features, mask = pad_sequence_to_max(feature_list)
        return {"features": features, "mask": mask}


def _sample_frame_indices_at_fps(
    total_frames: int, video_fps: float, sample_fps: float, max_frames: int
) -> list[int]:
    """Return frame indices sampled at sample_fps, capped at max_frames."""
    stride = video_fps / sample_fps
    n = min(int(total_frames / stride), max_frames)
    return [int(i * stride) for i in range(n)]
