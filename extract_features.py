"""One-time feature extraction script.

Downloads PerceptionTest videos from chancharikm/QualityCheck and extracts
Qwen3-VL visual features, saving them as .pt files for fast VAE training.

Usage:
    python extract_features.py \
        --cache_dir ./features_cache \
        --split train \
        --max_frames 8 \
        --model_name Qwen/Qwen3-VL-2B-Instruct \
        --device cuda \
        --max_samples -1   # -1 = all samples
"""

import argparse
import io
import os
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="./features_cache")
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", default="PerceptionTest")
    parser.add_argument("--dataset_name", default="chancharikm/QualityCheck")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 = all")
    return parser.parse_args()


def sample_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    if total_frames <= n_frames:
        return list(range(total_frames))
    step = total_frames / n_frames
    return [int(i * step) for i in range(n_frames)]


def extract_one(sample, backbone, processor, max_frames: int):
    """Extract Qwen3-VL features from a single dataset sample."""
    import decord
    from qwen_vl_utils import process_vision_info

    # Try common video field names
    video_bytes = None
    for field in ["video", "video_bytes", "mp4"]:
        if field in sample and sample[field] is not None:
            video_bytes = sample[field]
            break

    if video_bytes is None:
        return None, f"No video field. Keys: {list(sample.keys())}"

    try:
        if isinstance(video_bytes, bytes):
            video_io = io.BytesIO(video_bytes)
        else:
            video_io = video_bytes

        vr = decord.VideoReader(video_io, ctx=decord.cpu(0))
        total_frames = len(vr)
        indices = sample_frame_indices(total_frames, max_frames)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    except Exception as e:
        return None, f"Video decode error: {e}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe the video."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        videos=video_inputs,
        **video_kwargs,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values_videos"]
    grid_thw = inputs["video_grid_thw"]

    features, mask = backbone.extract_features(pixel_values, grid_thw)
    # features: [1, N, D], squeeze batch dim, save as float16
    return {
        "features": features[0].cpu().half(),
        "mask": mask[0].cpu(),
    }, None


def main():
    args = parse_args()

    from datasets import load_dataset
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
    ds = load_dataset(args.dataset_name, name=args.subset, split=args.split)

    n_total = len(ds) if args.max_samples == -1 else min(args.max_samples, len(ds))
    print(f"Processing {n_total} samples → {out_dir}")

    n_saved = 0
    n_errors = 0

    for i in tqdm(range(n_total), desc="Extracting features"):
        out_path = out_dir / f"{i:06d}.pt"
        if out_path.exists():
            n_saved += 1
            continue  # resume support

        sample = ds[i]
        result, err = extract_one(sample, backbone, processor, args.max_frames)

        if err is not None:
            print(f"[WARN] Sample {i}: {err}")
            n_errors += 1
            continue

        torch.save(result, out_path)
        n_saved += 1

    print(f"\nDone. Saved: {n_saved}, Errors: {n_errors}")


if __name__ == "__main__":
    main()
