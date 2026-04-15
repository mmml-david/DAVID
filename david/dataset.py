"""Dataset for PerceptionTest videos from chancharikm/QualityCheck.

Supports two modes:
  - cached: Load pre-extracted .pt feature files (fast, recommended for training).
  - online: Stream videos from HuggingFace, extract frames, run through backbone.
"""

import os
import logging
import torch

logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)
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
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        self.mode = mode
        self.split = split
        self.sample_fps = sample_fps
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

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
        from .utils import open_video_reader, sample_frame_indices_at_fps

        sample = self.dataset[idx]

        video_source = next(
            (sample[f] for f in ["video", "video_bytes", "mp4"] if f in sample and sample[f] is not None),
            None,
        )
        if video_source is None:
            raise ValueError(f"No video field found in sample {idx}. Keys: {list(sample.keys())}")

        vr = open_video_reader(video_source)
        indices = sample_frame_indices_at_fps(len(vr), vr.get_avg_fps(), self.sample_fps, self.max_frames)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]

        features, mask = self.backbone.extract_features_from_frames(
            frames, self.sample_fps, self.min_pixels, self.max_pixels
        )
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
