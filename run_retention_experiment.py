#!/usr/bin/env python
"""
Run retention-rate ablation: evaluate Qwen and DAVID at 25%, 50%, 75%, 100%
visual token retention, then plot accuracy vs retention rate.

Usage:
python run_retention_experiment.py \
    --vae_checkpoint checkpoints/step_0000650.pt \
    --video_root /DATA/dataset/PerceptionTest
python run_retention_experiment.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --vae_config configs/train_config.yaml \
    --vae_checkpoint checkpoints/8b_step_0001000.pt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def run_eval(method, ratio, args):
    """Run evaluate_vqa.py and return accuracy directly by parsing the output summary."""
    cmd = [
        sys.executable, "evaluate_vqa.py",
        "--questions_json", args.questions_json,
        "--method", method,
        "--max_samples", str(args.max_samples),
        "--model_name", args.model_name,
        "--vae_config", args.vae_config,
        "--vae_checkpoint", args.vae_checkpoint,
        "--video_root", args.video_root,
        "--output_dir", args.output_dir,
    ]
    # Always pass ratios so they're recorded in the summary JSON
    cmd += ["--visual_prefix_ratio", str(ratio)]
    if method == "david":
        cmd += ["--vae_prefix_ratio", str(ratio)]

    print(f"\n{'='*60}")
    print(f"Running: {method} @ {ratio*100:.0f}% retention")
    print(f"{'='*60}", flush=True)
    # Stream output live so user sees progress bars
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    summary_path = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if "Saved summary:" in line:
            summary_path = line.split("Saved summary:")[-1].strip()
    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] {method}@{ratio} failed with return code {proc.returncode}")
        return None

    if summary_path is None or not Path(summary_path).exists():
        print(f"[ERROR] Could not find summary file for {method}@{ratio}")
        return None

    with open(summary_path) as f:
        data = json.load(f)

    method_stats = data.get("method_stats", {})
    if method in method_stats:
        acc = method_stats[method].get("accuracy", 0.0)
    else:
        acc = 0.0

    print(f"  → {method}@{ratio*100:.0f}%: accuracy={acc:.4f}")
    return acc


def collect_from_existing(output_dir):
    """Scan existing summary files and build results dict from saved ratios."""
    results = {"qwen": {}, "david": {}}
    for p in Path(output_dir).glob("*_summary.json"):
        with open(p) as f:
            data = json.load(f)
        vpr = data.get("visual_prefix_ratio")
        ratio_key = str(vpr) if vpr is not None else "1.0"
        method_stats = data.get("method_stats", {})
        for method in ["qwen", "david"]:
            if method in method_stats and "accuracy" in method_stats[method]:
                results[method][ratio_key] = method_stats[method]["accuracy"]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_json", default="perception_test_mini100.json")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vae_config", default="configs/train_config_2b_online.yaml")
    parser.add_argument("--vae_checkpoint", default="checkpoints/step_0000650.pt")
    parser.add_argument("--video_root", default="/DATA/dataset/PerceptionTest")
    parser.add_argument("--output_dir", default="./eval_outputs/retention_experiment")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip eval, collect results from existing summaries and re-plot")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(args.output_dir) / "retention_results.json"

    if not args.plot_only:
        results = {"qwen": {}, "david": {}}

        for ratio in args.ratios:
            # Qwen baseline
            acc = run_eval("qwen", ratio, args)
            if acc is not None:
                results["qwen"][str(ratio)] = acc

            # DAVID
            acc = run_eval("david", ratio, args)
            if acc is not None:
                results["david"][str(ratio)] = acc

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    else:
        # Rebuild results from all existing summary files
        results = collect_from_existing(args.output_dir)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Collected results from existing summaries: {results_path}")

    # Print results table
    print(f"\n{'Retention':>10} {'Qwen':>10} {'DAVID':>10}")
    print("-" * 32)
    all_ratios = sorted(set(list(results["qwen"].keys()) + list(results["david"].keys())), key=float)
    for r in all_ratios:
        qwen_acc = results["qwen"].get(r, None)
        david_acc = results["david"].get(r, None)
        qwen_str = f"{qwen_acc*100:.1f}%" if qwen_acc is not None else "—"
        david_str = f"{david_acc*100:.1f}%" if david_acc is not None else "—"
        print(f"{float(r)*100:>9.0f}% {qwen_str:>10} {david_str:>10}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, label, color, marker in [
        ("qwen", "Qwen3-VL (baseline)", "#2196F3", "o"),
        ("david", "DAVID (ours)", "#FF5722", "s"),
    ]:
        ratios = sorted(results[method].keys(), key=float)
        if not ratios:
            continue
        x = [float(r) * 100 for r in ratios]
        y = [results[method][r] * 100 for r in ratios]
        ax.plot(x, y, marker=marker, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Retention Rate (%)", fontsize=13)
    ax.set_ylabel("VQA Accuracy (%)", fontsize=13)
    ax.set_title("VQA Accuracy vs Visual Token Retention Rate", fontsize=14)
    ax.set_xticks([25, 50, 75, 100])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 105)

    plot_path = Path(args.output_dir) / "retention_accuracy.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
