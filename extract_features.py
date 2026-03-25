"""One-time feature extraction script.

Downloads PerceptionTest videos from chancharikm/QualityCheck and extracts
Qwen3-VL visual features, saving them as .pt files for fast VAE training.

Usage:
    python extract_features.py \
        --cache_dir ./features_cache \
        --split train \
        --max_frames 8 \
        --model_name Qwen/Qwen3-VL-8B-Instruct \
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
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
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
                vr = decord.VideoReader(video_source["path"], ctx=decord.cpu(0))
            elif video_source.get("bytes") is not None:
                vr = decord.VideoReader(io.BytesIO(video_source["bytes"]), ctx=decord.cpu(0))
            else:
                return None, "Video dict has neither 'path' nor 'bytes'"
        elif isinstance(video_source, bytes):
            vr = decord.VideoReader(io.BytesIO(video_source), ctx=decord.cpu(0))
        else:
            vr = decord.VideoReader(video_source, ctx=decord.cpu(0))

        total_frames = len(vr)
        indices = sample_frame_indices(total_frames, max_frames)
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
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe the video."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
        video_kwargs["fps"] = video_kwargs["fps"][0]

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
