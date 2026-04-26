"""
Attention column-entropy visualization: attention matrix (rows = key j, cols = query i),
with distribution plots showing early (flat) vs late (peaked) rows.
Mirrors the user's hand-drawn sketch.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

# ── Palette ───────────────────────────────────────────────────────────────────
C_EARLY   = "#4878CF"   # blue
C_LATE    = "#D65F5F"   # red-orange
C_MID     = "#6ACC65"   # green (unused but available)
C_ENTROPY = "#E67E22"
C_GRID    = "#DDDDDD"
C_TEXT    = "#222222"

N_K = 12    # DAVID token key positions j  (y-axis, top=early, bottom=late)
N_Q = 18    # query positions i  (x-axis)

# ── Build synthetic attention matrix  A[j, i] = p_{i|j} ─────────────────────
# Each row j is a probability distribution over i (the "column" of the true
# attention matrix, column-normalised).  Early rows are flat; late rows peaked.
def make_row(j, N_K, N_Q, seed):
    rng = np.random.default_rng(seed)
    frac = j / (N_K - 1)          # 0 = earliest, 1 = latest

    peak = int(N_Q * rng.uniform(0.2, 0.8))
    x = np.arange(N_Q)

    # Smooth continuous transition: concentration grows gently, noise fades gently
    conc  = 5.5 * (frac ** 1.1)          # ≈0 for early rows, ≈5.5 for latest
    noise = 0.55 - 0.40 * (frac ** 0.8)  # 0.55 for early, ≈0.15 for latest

    lg = -conc * (x - peak) ** 2 / (N_Q / 4) ** 2
    lg += rng.normal(0, noise, N_Q)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    return p

seeds = [7, 13, 31, 42, 57, 61, 74, 83, 97, 103, 117, 131]
A = np.array([make_row(j, N_K, N_Q, seeds[j]) for j in range(N_K)])

def h(p):
    p = np.clip(p, 1e-10, None)
    return -(p * np.log(p)).sum()

H = np.array([h(A[j]) for j in range(N_K)])
qs = np.arange(N_Q)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7.5), facecolor="white")
gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    left=0.10, right=0.97,
    top=0.88,  bottom=0.10,
    width_ratios=[3.2, 1.5, 1.0],
    height_ratios=[1, 0.22, 1],
    wspace=0.45, hspace=0.12,
)

ax_heat   = fig.add_subplot(gs[:, 0])       # main heatmap (spans all rows)
ax_early  = fig.add_subplot(gs[0, 1])       # early-token distribution
ax_spacer = fig.add_subplot(gs[1, 1])       # invisible spacer
ax_late   = fig.add_subplot(gs[2, 1])       # late-token distribution
ax_ent    = fig.add_subplot(gs[:, 2])       # entropy curve (vertical)

ax_spacer.set_visible(False)

# ── Main heatmap ──────────────────────────────────────────────────────────────
# Normalise each row to [0,1] for display so concentration is visually apparent
A_disp = A / A.max(axis=1, keepdims=True)

im = ax_heat.imshow(
    A_disp, aspect="auto", cmap="Blues",
    vmin=0, vmax=1, interpolation="gaussian",
    origin="upper",
)

# Subtle grid
for j in range(N_K + 1):
    ax_heat.axhline(j - 0.5, color=C_GRID, lw=0.5, zorder=3)
for i in range(N_Q + 1):
    ax_heat.axvline(i - 0.5, color=C_GRID, lw=0.5, zorder=3)

# Highlight the two example rows
EARLY_ROW, LATE_ROW = 1, N_K - 2
for row, colour, lbl in [(EARLY_ROW, C_EARLY, "early"), (LATE_ROW, C_LATE, "late")]:
    for spine_y in [row - 0.49, row + 0.49]:
        ax_heat.axhline(spine_y, color=colour, lw=2.5, zorder=4)
    ax_heat.axvline(-0.49, ymin=(N_K - 1 - row - 0.4) / N_K,
                    ymax=(N_K - 1 - row + 0.4) / N_K,
                    color=colour, lw=2.5, zorder=4, clip_on=False)

ax_heat.set_xlabel("Query position  $i$", fontsize=11, color=C_TEXT, labelpad=5)
ax_heat.set_yticks([EARLY_ROW, (N_K - 1) // 2, LATE_ROW])
ax_heat.set_yticklabels(["early\n(coarse)", "·\n·\n·", "late\n(fine)"],
                         fontsize=9.5, color=C_TEXT)
ax_heat.set_xticks([0, N_Q // 2, N_Q - 1])
ax_heat.set_xticklabels(["0", str(N_Q // 2), str(N_Q - 1)], fontsize=9)
ax_heat.tick_params(length=3, color="#888888")
for sp in ax_heat.spines.values():
    sp.set_color("#BBBBBB"); sp.set_linewidth(1.0)

# Y-axis label as a rotated text on the left
ax_heat.text(-0.15, 0.5, "DAVID token  /  key position  $j$",
             transform=ax_heat.transAxes,
             rotation=90, ha="center", va="center",
             fontsize=10.5, color=C_TEXT)

# ── Helper to style distribution axes ────────────────────────────────────────
def style_dist(ax):
    ax.set_facecolor("#F8FAFD")
    ax.grid(axis="y", color=C_GRID, linewidth=0.6)
    ax.grid(axis="x", color=C_GRID, linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC"); sp.set_linewidth(0.8)
    ax.tick_params(labelsize=8, length=2, colors="#555555")
    ax.set_xlim(-0.5, N_Q - 0.5)

# ── Early distribution panel ──────────────────────────────────────────────────
style_dist(ax_early)
p_early = A[EARLY_ROW]
ax_early.bar(qs, p_early, color=C_EARLY, alpha=0.80, width=0.85, zorder=2)
ax_early.fill_between(qs, p_early, alpha=0.25, color=C_EARLY, step="mid")
ax_early.set_xticks([]); ax_early.set_yticks([])
ax_early.set_title("Early token  $j \\approx 0$", fontsize=9.5,
                    fontweight="bold", color=C_EARLY, pad=4)
ax_early.text(0.97, 0.90,
              f"$H = {h(p_early):.2f}$ nats\n(high entropy)",
              transform=ax_early.transAxes,
              ha="right", va="top", fontsize=8, color=C_EARLY,
              bbox=dict(facecolor="white", edgecolor=C_EARLY,
                        linewidth=0.8, boxstyle="round,pad=0.3", alpha=0.9))
ax_early.set_ylabel("$p_{\\cdot j}$", fontsize=9, color=C_TEXT, labelpad=2)

# ── Late distribution panel ───────────────────────────────────────────────────
style_dist(ax_late)
p_late = A[LATE_ROW]
ax_late.bar(qs, p_late, color=C_LATE, alpha=0.80, width=0.85, zorder=2)
ax_late.fill_between(qs, p_late, alpha=0.25, color=C_LATE, step="mid")
ax_late.set_xticks([0, N_Q // 2, N_Q - 1])
ax_late.set_xticklabels(["0", str(N_Q // 2), str(N_Q - 1)], fontsize=8)
ax_late.set_yticks([])
ax_late.set_title("Late token  $j \\approx N{-}1$", fontsize=9.5,
                   fontweight="bold", color=C_LATE, pad=4)
ax_late.set_xlabel("Query position  $i$", fontsize=8.5, color=C_TEXT, labelpad=3)
ax_late.text(0.97, 0.90,
             f"$H = {h(p_late):.2f}$ nats\n(lower entropy)",
             transform=ax_late.transAxes,
             ha="right", va="top", fontsize=8, color=C_LATE,
             bbox=dict(facecolor="white", edgecolor=C_LATE,
                       linewidth=0.8, boxstyle="round,pad=0.3", alpha=0.9))
ax_late.set_ylabel("$p_{\\cdot j}$", fontsize=9, color=C_TEXT, labelpad=2)

# ── Entropy curve (vertical, H_j vs j) ───────────────────────────────────────
ax_ent.set_facecolor("#F8FAFD")
for sp in ax_ent.spines.values():
    sp.set_color("#CCCCCC"); sp.set_linewidth(0.8)
ax_ent.tick_params(labelsize=8, length=2, colors="#555555")
ax_ent.grid(axis="x", color=C_GRID, linewidth=0.5)

js_plot = np.arange(N_K)
# Plot H_j horizontally (x=H, y=j) so y aligns with the heatmap rows
ax_ent.plot(H, js_plot, "o-", color=C_ENTROPY, linewidth=2.0,
            markersize=5, alpha=0.9, zorder=3)
ax_ent.fill_betweenx(js_plot, H, color=C_ENTROPY, alpha=0.15)

# Mark the two example rows
ax_ent.scatter([H[EARLY_ROW]], [EARLY_ROW], color=C_EARLY, s=70, zorder=5,
               edgecolors="white", linewidth=1.2)
ax_ent.scatter([H[LATE_ROW]],  [LATE_ROW],  color=C_LATE,  s=70, zorder=5,
               edgecolors="white", linewidth=1.2)

ax_ent.invert_yaxis()   # top = j=0 (early), bottom = j=N-1 (late)
ax_ent.set_ylim(N_K - 0.5, -0.5)
ax_ent.set_xlabel("$H(p_{\\cdot j})$\n[nats]", fontsize=9, color=C_TEXT, labelpad=3)
ax_ent.set_title("Column\nEntropy", fontsize=9, fontweight="bold",
                 color=C_ENTROPY, pad=4)
ax_ent.set_yticks([])
ax_ent.set_xlim(left=0)

# Gradient annotation arrow
ax_ent.annotate("entropy\ndecreases", xy=(H[LATE_ROW] + 0.02, LATE_ROW - 0.3),
                xytext=(H[LATE_ROW] + 0.02, EARLY_ROW + 0.5),
                fontsize=7.5, ha="left", color=C_ENTROPY,
                arrowprops=dict(arrowstyle="-|>", color=C_ENTROPY, lw=1.3))

# ── Title & formula ───────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         "Attention Column-Entropy Regularization",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")

fig.text(0.5, 0.924,
         r"$\mathcal{L}_{\mathrm{ent}} = \lambda \sum_{j} "
         r"\frac{j}{N} \cdot H(p_{\cdot j})$"
         r"     ·     early tokens may be diffuse;  later tokens are penalized for broad influence",
         ha="center", va="top", fontsize=10, color="#444444")

# ── Connecting lines from heatmap rows to distribution panels ─────────────────
# These are drawn in figure coordinates
for row, ax_dist, colour in [(EARLY_ROW, ax_early, C_EARLY),
                              (LATE_ROW,  ax_late,  C_LATE)]:
    # Get position of heatmap right edge at this row
    row_frac_heat = 1 - (row + 0.5) / N_K   # fraction from bottom of axes
    # Left edge of distribution panel
    heat_bbox = ax_heat.get_position()
    dist_bbox = ax_dist.get_position()

    y_heat = heat_bbox.y0 + row_frac_heat * heat_bbox.height
    y_dist = dist_bbox.y0 + 0.5 * dist_bbox.height

    fig.add_artist(plt.matplotlib.lines.Line2D(
        [heat_bbox.x1 + 0.005, dist_bbox.x0 - 0.005],
        [y_heat, y_dist],
        transform=fig.transFigure,
        color=colour, linewidth=1.5, linestyle="--", alpha=0.7, zorder=10,
    ))

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = "/Users/ethan/ethanfolder/cmu/mmml/project/DAVID/scripts/attn_matrix_viz.pdf"
out_png = out_pdf.replace(".pdf", ".png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=220)
fig.savefig(out_png, bbox_inches="tight", dpi=220)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
