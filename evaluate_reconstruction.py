"""Evaluate DAVID VAE reconstruction quality on cached features.

Computes per-sample and aggregate metrics:
  - L1 (mean absolute error)
  - MSE (mean squared error)
  - cosine similarity
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from david.dataset import PerceptionTestVideoDataset
from david.vae import DAVIDConfig, DAVIDVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DAVID reconstruction metrics on cached features")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint (.pt)")
    parser.add_argument("--feature_cache_dir", default=None, help="Override cfg.data.feature_cache_dir")
    parser.add_argument("--split", default="validation", help="Dataset split under feature_cache_dir")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', 'cpu', etc.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument(
        "--questions_json",
        default=None,
        help="Optional JSON question file (e.g. perception_test_mini100.json). "
             "If set, evaluate only videos referenced by video_url/sample_id.",
    )
    parser.add_argument("--output_dir", default="./eval_outputs")
    parser.add_argument("--debug_log", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_vae_from_checkpoint(config_path: str, checkpoint_path: str, device: torch.device) -> DAVIDVAE:
    cfg = load_yaml(config_path)
    david_cfg = DAVIDConfig.from_dict(dict(cfg.get("model", {})))
    vae = DAVIDVAE(david_cfg).to(device)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    vae.load_state_dict(state_dict, strict=True)
    vae.eval()
    return vae


def per_sample_metrics(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
    """Compute L1/MSE/CosSim on valid tokens only for one sample."""
    valid = mask.bool()
    if valid.sum().item() == 0:
        return 0.0, 0.0, 0.0

    x = target[valid]  # [Nv, D]
    y = recon[valid]   # [Nv, D]

    l1 = torch.abs(y - x).mean().item()
    mse = torch.square(y - x).mean().item()
    cos = torch.nn.functional.cosine_similarity(
        y.reshape(1, -1),
        x.reshape(1, -1),
        dim=1,
        eps=1e-8,
    ).item()
    return float(l1), float(mse), float(cos)


def video_names_from_questions_json(json_path: str) -> list[str]:
    """Extract unique video filename stems from a question JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    names: list[str] = []
    seen: set[str] = set()
    for entry in data:
        video_name = None
        video_url = entry.get("video_url", "")
        if isinstance(video_url, str) and video_url:
            video_name = os.path.splitext(os.path.basename(video_url))[0]
        if not video_name:
            sample_id = entry.get("sample_id", "")
            if isinstance(sample_id, str) and sample_id:
                video_name = sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id
        if video_name and video_name not in seen:
            seen.add(video_name)
            names.append(video_name)
    return names


def main() -> None:
    args = parse_args()

    def dbg(msg: str):
        if args.debug_log:
            print(f"[Debug] {msg}", flush=True)

    cfg = load_yaml(args.config)
    feature_cache_dir = args.feature_cache_dir or cfg["data"]["feature_cache_dir"]
    device = resolve_device(args.device)

    dbg(f"device={device}")
    dbg(f"feature_cache_dir={feature_cache_dir}, split={args.split}")

    dataset = PerceptionTestVideoDataset(
        feature_cache_dir=feature_cache_dir,
        split=args.split,
        mode="cached",
    )
    selected_video_names: list[str] | None = None
    matched_video_names: list[str] = []
    missing_video_names: list[str] = []

    if args.questions_json:
        selected_video_names = video_names_from_questions_json(args.questions_json)
        wanted = set(selected_video_names)
        dbg(f"questions_json={args.questions_json} unique_videos={len(wanted)}")

        file_stems = [p.stem for p in dataset.feature_files]
        stem_to_idx = {stem: i for i, stem in enumerate(file_stems)}
        matched_indices = [stem_to_idx[name] for name in selected_video_names if name in stem_to_idx]
        matched_video_names = [name for name in selected_video_names if name in stem_to_idx]
        missing_video_names = [name for name in selected_video_names if name not in stem_to_idx]

        from torch.utils.data import Subset

        dataset = Subset(dataset, matched_indices)
        print(
            f"Questions subset: requested={len(selected_video_names)} matched={len(matched_indices)} "
            f"missing={len(missing_video_names)}"
        )
        if missing_video_names and args.debug_log:
            print(f"[Debug] missing video ids (first 20): {missing_video_names[:20]}", flush=True)

    if args.max_samples > 0:
        from torch.utils.data import Subset

        n = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        dbg(f"using subset with {n} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PerceptionTestVideoDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    vae = load_vae_from_checkpoint(args.config, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Evaluating split='{args.split}' on {len(dataset)} samples")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"recon_metrics_{args.split}_{ts}"
    jsonl_path = out_dir / f"{base}.jsonl"
    summary_path = out_dir / f"{base}_summary.json"

    m_ratios = [0.25, 0.5, 0.75, 1.0]
    per_sample: list[dict[str, Any]] = []
    ratio_stats: dict[float, dict[str, float]] = {
        r: {"l1": 0.0, "mse": 0.0, "cos": 0.0, "count": 0.0}
        for r in m_ratios
    }
    n_samples = 0

    pbar = tqdm(loader, desc="Reconstruction eval")
    with torch.no_grad():
        for batch in pbar:
            features = batch["features"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device)

            for i in range(features.shape[0]):
                sample_features = features[i:i + 1]
                sample_mask = mask[i:i + 1]
                n_valid_tokens = int(sample_mask[0].sum().item())

                if n_valid_tokens <= 0:
                    n_samples += 1
                    continue

                for ratio in m_ratios:
                    m_tokens = max(1, min(n_valid_tokens, int(round(n_valid_tokens * ratio))))
                    output = vae(sample_features, sample_mask, training=False, m=m_tokens)
                    l1, mse, cos = per_sample_metrics(output.recon[0], sample_features[0], sample_mask[0])
                    rec = {
                        "sample_index": n_samples,
                        "video_name": matched_video_names[n_samples] if n_samples < len(matched_video_names) else None,
                        "m_ratio": ratio,
                        "m_tokens": m_tokens,
                        "l1": l1,
                        "mse": mse,
                        "cosine_similarity": cos,
                        "n_valid_tokens": n_valid_tokens,
                    }
                    per_sample.append(rec)
                    ratio_stats[ratio]["l1"] += l1
                    ratio_stats[ratio]["mse"] += mse
                    ratio_stats[ratio]["cos"] += cos
                    ratio_stats[ratio]["count"] += 1.0

                n_samples += 1

            count_full = ratio_stats[1.0]["count"]
            if count_full > 0:
                pbar.set_postfix(
                    ratio="1.0",
                    l1=f"{(ratio_stats[1.0]['l1'] / count_full):.6f}",
                    mse=f"{(ratio_stats[1.0]['mse'] / count_full):.6f}",
                    cos=f"{(ratio_stats[1.0]['cos'] / count_full):.6f}",
                )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in per_sample:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    metrics_by_ratio: dict[str, dict[str, float | None]] = {}
    for ratio in m_ratios:
        count = ratio_stats[ratio]["count"]
        metrics_by_ratio[f"{ratio:.2f}"] = {
            "l1_mean": (ratio_stats[ratio]["l1"] / count) if count > 0 else None,
            "mse_mean": (ratio_stats[ratio]["mse"] / count) if count > 0 else None,
            "cosine_similarity_mean": (ratio_stats[ratio]["cos"] / count) if count > 0 else None,
        }

    summary = {
        "timestamp": ts,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "feature_cache_dir": feature_cache_dir,
        "split": args.split,
        "questions_json": args.questions_json,
        "device": str(device),
        "num_samples": n_samples,
        "m_ratios": m_ratios,
        "selection": {
            "requested_videos": len(selected_video_names) if selected_video_names is not None else None,
            "matched_videos": len(matched_video_names) if selected_video_names is not None else None,
            "missing_videos": len(missing_video_names) if selected_video_names is not None else None,
        },
        "metrics_by_ratio": metrics_by_ratio,
        "outputs": {
            "jsonl": str(jsonl_path),
            "summary_json": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"Saved per-sample metrics: {jsonl_path}")
    print(f"Saved summary: {summary_path}")
    if n_samples > 0:
        print("Aggregate metrics by m_ratio:")
        for ratio in m_ratios:
            rkey = f"{ratio:.2f}"
            m = summary["metrics_by_ratio"][rkey]
            if m["l1_mean"] is None:
                continue
            print(
                f"  m_ratio={rkey}: "
                f"L1={m['l1_mean']:.6f}, "
                f"MSE={m['mse_mean']:.6f}, "
                f"CosSim={m['cosine_similarity_mean']:.6f}"
            )


if __name__ == "__main__":
    main()
