"""DAVID VAE training script.

Usage:
    # Cached mode (after running extract_features.py):
    python train.py --config configs/train_config.yaml

    # Online mode (extracts features during training, slower):
    python train.py --config configs/train_config.yaml --online

    # Smoke test (1 batch, 2 steps, no WandB):
    python train.py --config configs/train_config.yaml --smoke_test
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─── Config helpers ──────────────────────────────────────────────────────────

class DotDict(dict):
    """Dict subclass with attribute-style access."""
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str) -> DotDict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DotDict(raw)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(vae, optimizer, ema, step: int, loss: float, checkpoint_dir: str):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / f"step_{step:07d}.pt"
    torch.save({
        "step": step,
        "loss": loss,
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict": ema.state_dict(),
    }, path)
    print(f"  [Checkpoint] Saved to {path}")


def load_checkpoint(vae, optimizer, ema, checkpoint_dir: str):
    """Resume from the latest checkpoint in checkpoint_dir. Returns starting step."""
    ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    if not ckpts:
        return 0
    latest = ckpts[-1]
    state = torch.load(latest, map_location="cpu", weights_only=False)
    vae.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if "ema_state_dict" in state:
        ema.load_state_dict(state["ema_state_dict"])
    print(f"  [Checkpoint] Resumed from {latest} (step {state['step']})")
    return state["step"]


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--online", action="store_true", help="Extract features online (no cache)")
    parser.add_argument("--smoke_test", action="store_true", help="Quick sanity check (2 steps)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda, mps)")
    parser.add_argument("--debug_log", action="store_true", help="Print concise progress logs")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    def dbg(msg: str):
        if args.debug_log:
            print(f"[Debug] {msg}", flush=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Imports (deferred so smoke tests don't require full deps) ──
    from david.vae import DAVIDVAE, DAVIDConfig
    from david.loss import david_loss, BetaScheduler
    from david.dataset import PerceptionTestVideoDataset
    from david.utils import EMAModel

    # ── Model config ──
    david_cfg = DAVIDConfig.from_dict(dict(cfg.model))
    vae = DAVIDVAE(david_cfg).to(device)
    n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"DAVID VAE parameters: {n_params:,}")

    ema = EMAModel(vae, decay=cfg.training.ema_decay)

    # ── Optimizer ──
    optimizer = AdamW(
        vae.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.max_steps)
    beta_sched = BetaScheduler(
        beta_target=cfg.training.beta_target,
        warmup_start=cfg.training.warmup_steps,
        warmup_end=cfg.training.warmup_steps + cfg.training.beta_warmup_steps,
    )

    # ── Dataset ──
    if args.smoke_test:
        from torch.utils.data import TensorDataset
        D = david_cfg.input_dim
        N = 16  # synthetic sequence length
        B_total = cfg.training.batch_size * 2
        fake_features = torch.randn(B_total, N, D)
        fake_mask = torch.ones(B_total, N, dtype=torch.bool)
        dataset = TensorDataset(fake_features, fake_mask)
    elif args.online:
        from david.backbone import Qwen3VLBackbone
        backbone = Qwen3VLBackbone(
            model_name=cfg.model.backbone_name,
            device=device,
            dtype=torch.bfloat16,
        )
        dataset = PerceptionTestVideoDataset(
            feature_cache_dir=cfg.data.feature_cache_dir,
            split="train",
            mode="online",
            hf_dataset_name=cfg.data.dataset_name,
            subset=cfg.data.subset,
            backbone=backbone,
            sample_fps=cfg.data.sample_fps,
            max_frames=cfg.data.max_frames,
        )
    else:
        dataset = PerceptionTestVideoDataset(
            feature_cache_dir=cfg.data.feature_cache_dir,
            split="train",
            mode="cached",
        )
    dbg(f"Dataset ready: size={len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=None if args.smoke_test else PerceptionTestVideoDataset.collate_fn,
        num_workers=0 if args.smoke_test else cfg.data.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    dbg(f"DataLoader ready: batch_size={cfg.training.batch_size}, num_workers={0 if args.smoke_test else cfg.data.num_workers}")

    # ── WandB ──
    use_wandb = cfg.logging.use_wandb and not args.smoke_test
    if use_wandb:
        import wandb
        wandb.init(
            entity=cfg.logging.entity,
            project=cfg.logging.project,
            name=cfg.logging.get("run_name") or None,
            config=dict(cfg),
        )

    # ── Resume ──
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(vae, optimizer, ema, cfg.logging.checkpoint_dir)

    # ── Training loop ──
    max_steps = 2 if args.smoke_test else cfg.training.max_steps
    grad_accum = cfg.training.gradient_accumulation_steps

    step = start_step
    optimizer.zero_grad()
    vae.train()

    print(f"Starting training from step {start_step}")
    dbg("Entering training loop")

    with tqdm(total=max_steps, initial=start_step, desc="Training") as pbar:
        while step < max_steps:
            dbg("Starting new dataloader pass")
            loader_iter = iter(loader)
            while step < max_steps:
                if step >= max_steps:
                    break

                dbg(f"Fetching batch for step {step}")
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    dbg("Reached end of dataloader pass")
                    break
                dbg("Batch fetched")

                if isinstance(batch, (list, tuple)):
                    features, mask = batch[0].to(device), batch[1].to(device)
                else:
                    features, mask = batch["features"].to(device), batch["mask"].to(device)
                feature_dim = features.shape[-1]
                if feature_dim != david_cfg.input_dim:
                    raise ValueError(
                        f"Feature dim mismatch: batch has D={feature_dim}, "
                        f"but model.input_dim={david_cfg.input_dim}. "
                        "This usually means cached features were extracted with a different backbone."
                    )

                # Forward through VAE
                dbg("Running model forward")
                output = vae(features, mask, training=True)

                # Compute loss (reconstruct original features, masked for padding)
                dbg("Computing loss")
                beta = beta_sched.get_beta(step)
                loss_out = david_loss(
                    recon=output.recon,
                    target=features,
                    mu=output.mu,
                    logvar=output.logvar,
                    beta=beta,
                    m=output.m,
                )

                # Backward (scaled for gradient accumulation)
                dbg("Running backward")
                scaled_loss = loss_out.total / grad_accum
                scaled_loss.backward()

                if (step + 1) % grad_accum == 0:
                    dbg("Running optimizer step")
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.training.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    ema.update(vae)

                # Logging
                if step % cfg.logging.log_every == 0:
                    metrics = {
                        "loss/total": loss_out.total.item(),
                        "loss/mse": loss_out.mse,
                        "loss/kl": loss_out.kl,
                        "beta": beta,
                        "truncation/m": output.m,
                        "lr": scheduler.get_last_lr()[0],
                        "ema_decay": ema.decay,
                    }
                    if use_wandb:
                        wandb.log(metrics, step=step)
                    pbar.set_postfix({
                        "mse": f"{loss_out.mse:.4f}",
                        "kl": f"{loss_out.kl:.4f}",
                        "m": output.m,
                        "beta": f"{beta:.2e}",
                    })

                    if args.smoke_test:
                        print(f"\n[Smoke test] Step {step}: {metrics}")

                # Checkpointing
                if not args.smoke_test and step > 0 and step % cfg.logging.save_every == 0:
                    save_checkpoint(vae, optimizer, ema, step, loss_out.total.item(),
                                    cfg.logging.checkpoint_dir)

                step += 1
                pbar.update(1)

    if args.smoke_test:
        print("\n[Smoke test] PASSED — all shapes verified, loss computed successfully.")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
