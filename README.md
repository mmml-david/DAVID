# DAVID — Dynamic Adaptive Video Inference and Decoding

A hierarchical video representation that encodes a video as a prefix-ordered sequence of semantic tokens, built on top of a frozen Qwen3-VL vision backbone.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Workflow

### 1. Extract features (one-time)

Run all videos through the frozen Qwen3-VL backbone and cache features as `.pt` files.
All options are read from `configs/train_config.yaml` (`extraction:` section); CLI flags override.

```bash
# Single machine — all settings from config
python extract_features.py --config configs/train_config.yaml

# Override specific settings
python extract_features.py --config configs/train_config.yaml \
    --device cuda:1 --max_samples 100

# Distributed — run on each machine with a unique --shard_id
# (num_shards is set in the config; shard_id must be provided explicitly)
python extract_features.py --config configs/train_config.yaml --shard_id 0
python extract_features.py --config configs/train_config.yaml --shard_id 1
# ...
```

The script deduplicates by video filename stem (derived from the `video` column path).
Output files are named `{video_name}.pt`. Re-runs skip already-extracted files, so it
is safe to resume or re-run on any shard.

**Disk usage:** ~35 GB for the full PerceptionTest set (float16, ~11.6k videos).

**HuggingFace cache setup:** The dataset stores videos as individual files in the hub
snapshot. If the snapshot symlink tree is incomplete (e.g. interrupted download),
the extraction script will automatically recreate missing symlinks from already-present
local blobs with no network I/O. Videos whose blobs are absent will be skipped with a
warning; re-run after completing the download to fill gaps.

### 2. Smoke test

Verify the full pipeline using synthetic data (no backbone or cache needed):

```bash
python train.py --config configs/train_config.yaml --smoke_test
```

### 3. Train

```bash
<<<<<<< HEAD
# Single GPU
python train.py --config configs/train_config.yaml

# Multi-GPU via torchrun
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml
=======
# Single GPU (cached features)
python train.py --config configs/train_config.yaml

# Multi-GPU via torchrun
torchrun --nproc_per_node=2 train.py --config configs/train_config.yaml
>>>>>>> c25f7cc (✨ [Add] new subset of validation set)
```

Resume from the latest checkpoint:

```bash
python train.py --config configs/train_config.yaml --resume
```

Train in online mode (extracts features on-the-fly — slower, no cache needed):

```bash
python train.py --config configs/train_config.yaml --online
```

<<<<<<< HEAD
Override the WandB run name:

```bash
python train.py --config configs/train_config.yaml --run_name my-experiment
```

For the Qwen3-VL-2B model (lower VRAM, online mode):

```bash
python train.py --config configs/train_config_2b_online.yaml --online
```

### 4. Evaluate VQA (Qwen3-VL and DAVID+VAE)

Evaluate multiple-choice VQA accuracy on the PerceptionTest validation set using per-sample questions from a JSON file:
=======
For the 2B backbone, use the corresponding config:
>>>>>>> c25f7cc (✨ [Add] new subset of validation set)

```bash
python train.py --config configs/train_config_2b_online.yaml
```

### 4. Evaluate VQA (Qwen3-VL and DAVID+VAE)

Evaluate multiple-choice VQA accuracy on the PerceptionTest validation set.
The evaluation uses per-sample questions from a JSON file (`perception_test.json` for the full set, or `perception_test_mini100.json` for a stratified 100-sample subset).

```bash
# Baseline Qwen3-VL only (mini validation set, 100 samples)
python evaluate_vqa.py \
    --questions_json perception_test_mini100.json \
    --method qwen \
    --max_samples 0

# With a smaller model
python evaluate_vqa.py \
    --questions_json perception_test_mini100.json \
    --method qwen \
    --max_samples 0 \
    --model_name Qwen/Qwen3-VL-2B-Instruct

# 8B model on full validation set
python evaluate_vqa.py \
    --questions_json perception_test.json \
    --method qwen \
    --max_samples 0 \
    --model_name Qwen/Qwen3-VL-8B-Instruct

# Side-by-side comparison (Qwen3-VL + DAVID reconstruction)
# With random VAE (pipeline test, no checkpoint needed)
python evaluate_vqa.py \
    --questions_json perception_test_mini100.json \
    --method both \
    --max_samples 10

# DAVID evaluation with trained VAE (2B model)
python evaluate_vqa.py \
    --questions_json perception_test_mini100.json \
    --method david \
    --max_samples 0 \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --vae_checkpoint ./checkpoints/step_0000500.pt \
    --vae_config configs/train_config_2b_online.yaml
```

**Key flags:**

| Flag | Description |
|---|---|
| `--questions_json` | Path to JSON with per-sample questions/options/answers. Filters to valid set and resolves video URLs to local HF cache. |
| `--method` | `qwen` (baseline only), `david` (VAE reconstruction only), or `both` (side-by-side). |
| `--max_samples 0` | Evaluate all available samples (default 100). |
| `--model_name` | Backbone model (default `Qwen/Qwen3-VL-8B-Instruct`). |
| `--vae_checkpoint` | Path to trained VAE `.pt` file. Omit to use a randomly initialized VAE (for pipeline testing). |
| `--vae_config` | VAE config file. Must match the backbone's embedding dim (4096 for 8B, 2048 for 2B). |
| `--vae_device` | Device for VAE (`auto`, `cpu`, `cuda:X`). Use `cpu` to avoid OOM with large backbones. |
| `--hf_cache_root` | Root of HF hub cache (default `/DATA/huggingface/hub`). |

Without `--questions_json`, the script falls back to single-prompt mode using HF dataset streaming.

**Accuracy matching:** The script matches model responses against the correct answer using multiple strategies:
1. Response starts with the correct letter (e.g. `"C"`, `"C. moving"`)
2. Response contains the correct letter as a standalone token (e.g. `"The answer is C"`)
3. Response contains the correct option text (e.g. `"static or shaking"`)

Per-sample results include a `matched` field in the JSONL output.
All outputs (JSONL predictions + summary JSON) are saved to `./eval_outputs/`.

**Question sets:**
- `perception_test.json` — full validation set (19,140 questions across 5,260 videos)
- `perception_test_mini100.json` — stratified mini set (100 questions from 100 unique videos, covering all 13 question categories)

## Project Structure

```
DAVID/
├── pyproject.toml
├── perception_test.json         # Per-sample VQA questions for evaluation
├── configs/
<<<<<<< HEAD
│   ├── train_config.yaml        # 8B model — cached feature training
│   └── train_config_2b_online.yaml  # 2B model — online mode, lower VRAM
├── david/
│   ├── backbone.py              # Frozen Qwen3-VL vision encoder (LLM weights discarded)
│   ├── vae.py                   # DAVIDEncoder + DAVIDDecoder + DAVIDVAE (self-attention)
│   ├── dataset.py               # Dataset loader (cached and online modes)
│   ├── loss.py                  # Adaptive recon loss + KL + beta warm-up scheduler
│   └── utils.py                 # Padding, masking, interpolation helpers + EMAModel
├── extract_features.py          # One-time offline feature extraction
├── evaluate_vqa.py              # VQA accuracy evaluation (Qwen3-VL and DAVID+VAE)
└── train.py                     # Main training entry point (single-GPU and DDP)
=======
│   ├── train_config.yaml              # Hyperparameters for 8B backbone
│   └── train_config_2b_online.yaml    # Hyperparameters for 2B backbone
├── david/
│   ├── backbone.py                    # Frozen Qwen3-VL vision encoder (LLM weights discarded)
│   ├── vae.py                         # DAVIDEncoder + DAVIDDecoder + DAVIDVAE
│   ├── dataset.py                     # Dataset loader (cached and online modes)
│   ├── loss.py                        # Adaptive recon loss + KL + beta warm-up scheduler
│   └── utils.py                       # Padding, masking, interpolation helpers
├── extract_features.py                # One-time offline feature extraction
├── train.py                           # Training entry point (single/multi-GPU)
├── evaluate_vqa.py                    # VQA evaluation (baseline + DAVID)
├── download_videos.py                 # Download dataset videos from HuggingFace
├── perception_test.json               # Full validation question set
└── perception_test_mini100.json       # Stratified 100-sample mini set
>>>>>>> c25f7cc (✨ [Add] new subset of validation set)
```

## Configuration

Two config files are provided for different backbone sizes:

| Config | Backbone | `input_dim` |
|---|---|---|
| `configs/train_config.yaml` | Qwen3-VL-8B-Instruct | 4096 |
| `configs/train_config_2b_online.yaml` | Qwen3-VL-2B-Instruct | 2048 |

Each config has five sections:

| Section | Purpose |
|---|---|
| `model` | Backbone name, `input_dim`, encoder/decoder layers, `progressive_ratio`, `grad_checkpoint` |
| `training` | Batch size, learning rate, beta schedule, gradient clipping, `ema_decay` |
| `data` | Dataset name, feature cache path, `sample_fps`, `max_frames`, `min_pixels`/`max_pixels` |
| `extraction` | All `extract_features.py` defaults (`num_shards`, `cache_dir`, `device`, etc.) |
| `logging` | WandB entity/project/`run_name`, checkpoint directory, `log_every`, `eval_every`, `max_val_samples`, `save_every` |

Two configs are provided:
- `configs/train_config.yaml` — Qwen3-VL-8B, cached feature training (requires extracted features)
- `configs/train_config_2b_online.yaml` — Qwen3-VL-2B, online mode with lower resolution (`min_pixels=16384`, `max_pixels=204800`)

## Architecture

**Backbone** (`backbone.py`): Loads only the Qwen3-VL vision encoder onto GPU and discards the LLM weights (~16 GB VRAM saving). Videos are sampled at 1 fps and processed as a whole, producing a flat `[N, D]` token sequence where N varies with video duration (D=4096 for 8B, D=2048 for 2B).

**DAVID VAE** (`vae.py`): All three stages — input, latent, output — share the same `[B, N, D]` shape.

- **Encoder**: self-attention blocks over the full video token sequence (padding masked). Optional `progressive_ratio` causes later tokens to attend to progressively fewer random peers, strengthening the coarse-to-fine ordering incentive. Outputs `mu, logvar` at `[B, N, D]`.
- **Truncation**: sample `m ~ Uniform(1, N)` during training; zero-pad z beyond position m before decoding. This forces early tokens to carry coarse information and later tokens to carry refinements.
- **Decoder**: self-attention blocks over the zero-padded z; outputs reconstructed features `[B, N, D]`.

**Loss** (`loss.py`): `recon_loss + beta * KL`
- `recon_loss` is adaptive: exponent `p` and scale `s` both equal `2 * sigmoid(m/N)`, smoothly transitioning from L1 (small prefix -> coarse reconstruction) to L2 (full prefix -> precise reconstruction).
- `beta` is linearly annealed from 0 to `beta_target` over `beta_warmup_steps` to prevent posterior collapse.

At inference, tokens can be truncated at any prefix length for adaptive reasoning under variable token budgets.

**Training infrastructure** (`train.py`): Single-GPU or multi-GPU via `torchrun` (DDP). `bfloat16` autocast wraps forward and loss. EMA (`decay=0.999`) tracks a shadow copy of VAE weights and is saved in checkpoints. Validation runs every `eval_every` steps (MSE only); `best.pt` is saved on improvement alongside the regular `step_N.pt` checkpoint.
