"""Visualization of attention column-entropy regularization for the DAVID poster."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SEED = 42

C_EARLY  = "#4C9BE8"   # blue  – early / without-reg
C_LATE   = "#E8734C"   # orange – late / with-reg
C_MID    = "#9B59B6"   # purple – mid token
C_WEIGHT = "#2ECC71"   # green – weight ceiling line
C_BG     = "#F8FAFD"
C_GRID   = "#DDE3ED"

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def column_entropy(col):
    col = np.clip(col, 1e-10, None)
    col = col / col.sum()
    return -(col * np.log(col)).sum()

N = 14

def make_attn(N, seed, with_reg: bool):
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N))
    for j in range(N):
        logits = rng.normal(0, 1, N)
        if with_reg:
            concentration = 1.5 + 4.5 * (j / (N - 1)) ** 1.3
            peak = max(1, min(N - 2, int(rng.integers(1, N - 1))))
            sharpening = -concentration * (np.arange(N) - peak) ** 2 / (N / 3.5) ** 2
            logits = logits * 0.25 + sharpening
        else:
            logits = logits * 0.5
        A[:, j] = softmax(logits)
    return A

A_before = make_attn(N, SEED,     with_reg=False)
A_after  = make_attn(N, SEED + 1, with_reg=True)

H_before = np.array([column_entropy(A_before[:, j]) for j in range(N)])
H_after  = np.array([column_entropy(A_after[:,  j]) for j in range(N)])
js = np.arange(N)
w  = js / N

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5), facecolor="white")

# Reserve top 18% for title + formula
gs = gridspec.GridSpec(
    2, 4,
    figure=fig,
    left=0.05, right=0.97,
    top=0.78, bottom=0.11,
    wspace=0.50, hspace=0.60,
)

ax_before = fig.add_subplot(gs[:, 0])
ax_after  = fig.add_subplot(gs[:, 1])
ax_Hj     = fig.add_subplot(gs[0, 2])
ax_wHj    = fig.add_subplot(gs[1, 2])
ax_col    = fig.add_subplot(gs[:, 3])

def style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC"); sp.set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors="#444444", length=3)

# ── Attention matrices ───────────────────────────────────────────────────────
for ax, A, title, tag in [
    (ax_before, A_before, "Without Entropy Reg", "(a)"),
    (ax_after,  A_after,  "With Entropy Reg",    "(b)"),
]:
    ax.imshow(A, aspect="auto", cmap="Blues", vmin=0, vmax=0.55,
              interpolation="nearest")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5, color="#222222")
    ax.set_xlabel("Key position  j  (DAVID token)", fontsize=8, labelpad=3)
    ax.set_ylabel("Query position  i", fontsize=8, labelpad=3)
    ax.set_xticks([0, N // 2 - 1, N - 1])
    ax.set_xticklabels(["0\n(coarse)", f"{N//2-1}", f"{N-1}\n(fine)"], fontsize=7)
    ax.set_yticks([0, N - 1])
    ax.set_yticklabels(["0", f"{N-1}"], fontsize=7)
    ax.text(0.04, 0.97, tag, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", color="#555555")

# Annotate two example columns on the "with reg" panel
for j, lbl, col, yoff in [(1, "early\n(coarse)", C_EARLY, N + 2.5),
                           (N - 2, "late\n(fine)", C_LATE, N + 2.5)]:
    ax_after.annotate(
        lbl, xy=(j, N - 0.5), xytext=(j, N + 1.8),
        xycoords="data", textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4),
        fontsize=7, ha="center", color=col, fontweight="bold",
        annotation_clip=False,
    )

# ── Panel (c): column entropy H_j ────────────────────────────────────────────
style_ax(ax_Hj)
bw = 0.38
ax_Hj.bar(js - bw/2, H_before, width=bw, color=C_EARLY, alpha=0.75,
          label="without reg", zorder=2)
ax_Hj.bar(js + bw/2, H_after,  width=bw, color=C_LATE,  alpha=0.80,
          label="with reg",    zorder=2)
ax_Hj.set_xlabel("Key position  j", fontsize=8)
ax_Hj.set_ylabel("H(p·ⱼ)  [nats]", fontsize=8)
ax_Hj.set_title("(c)  Column Entropy  H(p·ⱼ)", fontsize=9,
                fontweight="bold", color="#222222")
ax_Hj.legend(fontsize=7.5, framealpha=0.9, loc="upper right")
ax_Hj.set_xticks([0, N // 2, N - 1])
ax_Hj.set_xticklabels(["0\n(coarse)", str(N // 2), f"{N-1}\n(fine)"], fontsize=7)

# ── Panel (d): weighted penalty w_j * H_j ────────────────────────────────────
style_ax(ax_wHj)
ax_wHj.bar(js - bw/2, w * H_before, width=bw, color=C_EARLY, alpha=0.75,
           label="without reg", zorder=2)
ax_wHj.bar(js + bw/2, w * H_after,  width=bw, color=C_LATE,  alpha=0.80,
           label="with reg",    zorder=2)
ax_wHj.plot(js, w * np.log(N), "--", color=C_WEIGHT, linewidth=1.5,
            label="max  (j/N)·ln N", alpha=0.85, zorder=3)
ax_wHj.set_xlabel("Key position  j", fontsize=8)
ax_wHj.set_ylabel("(j/N)·H(p·ⱼ)", fontsize=8)
ax_wHj.set_title("(d)  Weighted Penalty  (j/N)·H(p·ⱼ)", fontsize=9,
                fontweight="bold", color="#222222")
ax_wHj.legend(fontsize=7, framealpha=0.9, loc="upper left")
ax_wHj.set_xticks([0, N // 2, N - 1])
ax_wHj.set_xticklabels(["0\n(coarse)", str(N // 2), f"{N-1}\n(fine)"], fontsize=7)

# ── Panel (e): column distribution examples ───────────────────────────────────
style_ax(ax_col)
qs = np.arange(N)
examples = [
    (1,      "early  (j=1,  coarse)", C_EARLY, "o"),
    (N // 2, f"mid    (j={N//2})",    C_MID,   "s"),
    (N - 2,  f"late   (j={N-2},  fine)", C_LATE, "^"),
]
for j, lbl, colour, marker in examples:
    ax_col.plot(qs, A_after[:, j], f"-{marker}", color=colour, label=lbl,
                linewidth=1.8, markersize=5, alpha=0.9, zorder=3)
ax_col.set_xlabel("Query position  i", fontsize=8)
ax_col.set_ylabel("p_{i,j}  (column distribution)", fontsize=8)
ax_col.set_title("(e)  Column Distributions\n(with entropy reg)", fontsize=9,
                fontweight="bold", color="#222222")
ax_col.legend(fontsize=7.5, framealpha=0.9, loc="upper left")
ax_col.set_facecolor(C_BG)
for sp in ax_col.spines.values():
    sp.set_color("#CCCCCC"); sp.set_linewidth(0.8)
ax_col.tick_params(labelsize=8, colors="#444444", length=3)
ax_col.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
ax_col.set_xticks([0, N // 2, N - 1])
ax_col.set_xticklabels(["0", str(N // 2), str(N - 1)], fontsize=7)

# ── Title (top of figure) ─────────────────────────────────────────────────────
fig.text(0.5, 0.985, "Attention Column-Entropy Regularization",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")

# Formula line
fig.text(0.5, 0.935,
         r"$\mathcal{L}_{\mathrm{entropy}}\;=\;\lambda\!\sum_{j=0}^{N-1}\frac{j}{N}\cdot H(p_{\cdot j})$",
         ha="center", va="top", fontsize=13, color="#2C2C6C")

# One-line intuition
fig.text(0.5, 0.870,
         r"$p_{\cdot j}$ = column-normalised attention for key $j$  ·  "
         r"Later tokens (larger $j$) are penalised more for diffuse influence",
         ha="center", va="top", fontsize=9, color="#555555", style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = "/Users/ethan/ethanfolder/cmu/mmml/project/DAVID/scripts/entropy_loss_viz.pdf"
out_png = out_pdf.replace(".pdf", ".png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
