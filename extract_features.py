"""One-time feature extraction script.

Downloads PerceptionTest videos from chancharikm/QualityCheck and extracts
Qwen3-VL visual features, saving them as .pt files for fast VAE training.

Output filenames use the global dataset index: {global_idx:07d}.pt
This ensures files from different machines never collide when writing to a
shared directory.

Single-machine usage:
    python extract_features.py \
        --cache_dir ./features_cache \
        --split train \
        --max_frames 8 \
        --model_name Qwen/Qwen3-VL-8B-Instruct \
        --device cuda

Distributed usage (run on each machine with a different --shard_id):
    # Machine 0 of 4:
    python extract_features.py --num_shards 4 --shard_id 0 --cache_dir /shared/features_cache
    # Machine 1 of 4:
    python extract_features.py --num_shards 4 --shard_id 1 --cache_dir /shared/features_cache
    # ...and so on.

Each machine writes non-overlapping files named by their global index.
Re-running is safe — existing files are skipped (resume support).
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
    # Distributed extraction: split the dataset into num_shards contiguous ranges.
    # Each machine runs with a unique shard_id in [0, num_shards).
    # Output files are named by global index so all machines share one directory safely.
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of machines processing in parallel")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Index of this machine (0-indexed, must be < num_shards)")
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

    assert 0 <= args.shard_id < args.num_shards, \
        f"shard_id={args.shard_id} must be in [0, num_shards={args.num_shards})"

    n_total = len(ds) if args.max_samples == -1 else min(args.max_samples, len(ds))

    # Compute the contiguous slice of global indices this machine owns.
    # Indices are assigned as evenly as possible; the last shard absorbs any remainder.
    shard_size = n_total // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = n_total if args.shard_id == args.num_shards - 1 else start_idx + shard_size
    my_indices = range(start_idx, end_idx)

    print(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"global indices [{start_idx}, {end_idx}) — {len(my_indices)} samples → {out_dir}"
    )

    n_saved = 0
    n_errors = 0

    for global_idx in tqdm(my_indices, desc=f"Shard {args.shard_id}"):
        # Filename is the global dataset index — unique across all machines.
        out_path = out_dir / f"{global_idx:07d}.pt"
        if out_path.exists():
            n_saved += 1
            continue  # resume support

        sample = ds[global_idx]
        result, err = extract_one(sample, backbone, processor, args.max_frames)

        if err is not None:
            print(f"[WARN] Sample {global_idx}: {err}")
            n_errors += 1
            continue

        torch.save(result, out_path)
        n_saved += 1

    print(f"\nShard {args.shard_id} done. Saved: {n_saved}, Errors: {n_errors}")


if __name__ == "__main__":
    main()
