#!/usr/bin/env python3
"""
Analyse the INT8 rank-correlation run (addresses Reviewer ZdDi) and emit the
paper figure + summary numbers.

Reads `export_corrected/int8_rank_correlation.csv` (produced by
`int8_rank_correlation.py`) and reports:

  (a) our_fp32 -> our_int8   : the *pure quantization* effect (same weights,
      same test set, only the PTQ step differs) -- the direct test of "does INT8
      PTQ preserve the accuracy ranking".
  (b) nats_cifar10 -> our_int8 : the *paper-relevant* chain -- does the tabulated
      NATS-Bench FP32 accuracy (which we pair with the measured hardware costs)
      still rank architectures the same way as the actually-deployed INT8 model?

For each we report Spearman rho, Kendall tau-b, and top-k overlap; plus the mean
/ median / max absolute accuracy change from quantization and a *proper* crater
count (genuine accuracy collapse from quantization, i.e. FP32 clearly above
chance but INT8 dropping >10 pp -- as opposed to architectures that are already
dead at chance in FP32, which quantization cannot make worse).

Spearman / Kendall are implemented in numpy so the script needs only
numpy + matplotlib (no scipy).

Usage:
    python int8_rank_analysis.py
    python int8_rank_analysis.py --csv export_corrected/int8_rank_correlation.csv \
        --out-fig figures_corrected/fig10_int8_rank --no-png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent

WONG = {
    "blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
    "pink": "#CC79A7", "yellow": "#E69F00", "sky": "#56B4E9",
    "black": "#222222", "gray": "#9A9A9A",
}

# An FP32 accuracy at/below this is at CIFAR-10 chance (~0.10): a degenerate
# (e.g. all-`none`) architecture that is already dead before quantization.
CHANCE = 0.15
CRATER_DROP = 0.10  # >10 pp INT8 drop from a non-degenerate FP32 = a real crater


# ------------------------------------------------------------- rank statistics
def _rankdata(a):
    """Average ranks, ties shared (like scipy.stats.rankdata 'average')."""
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def kendall_tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        dxi, dyi = x[i + 1:] - x[i], y[i + 1:] - y[i]
        sx, sy = np.sign(dxi), np.sign(dyi)
        prod = sx * sy
        conc += int((prod > 0).sum())
        disc += int((prod < 0).sum())
        tx += int(((sx == 0) & (sy != 0)).sum())
        ty += int(((sy == 0) & (sx != 0)).sum())
    denom = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / denom) if denom else float("nan")


def topk_overlap(idx, a, b, k):
    a, b = np.asarray(a), np.asarray(b)
    top_a = set(np.asarray(idx)[a.argsort()[::-1][:k]])
    top_b = set(np.asarray(idx)[b.argsort()[::-1][:k]])
    return len(top_a & top_b), k


def bootstrap_ci(x, y, fn, B=10000, seed=0, pct=(2.5, 97.5)):
    """Percentile bootstrap CI for a correlation statistic (resample archs)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        try:
            vals.append(fn(x[s], y[s]))
        except Exception:
            pass
    return tuple(float(v) for v in np.percentile(vals, pct))


# ----------------------------------------------------------------------- plot
def set_style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.labelpad": 3,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#CCCCCC",
        "grid.linewidth": 0.5, "grid.alpha": 0.6, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def make_figure(idx, nats, fp32, int8, degen_mask, stem: Path, also_png: bool):
    set_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    # ---- Panel A: pure quantization effect, our FP32 vs our INT8 -------------
    lo = min(fp32.min(), int8.min()) - 0.02
    hi = max(fp32.max(), int8.max()) + 0.02
    axL.plot([lo, hi], [lo, hi], color=WONG["gray"], lw=0.9, ls="--", zorder=1)
    axL.scatter(fp32[~degen_mask] * 100, int8[~degen_mask] * 100, s=16,
                color=WONG["blue"], alpha=0.75, edgecolor="none", zorder=3,
                label="architecture")
    if degen_mask.any():
        axL.scatter(fp32[degen_mask] * 100, int8[degen_mask] * 100, s=20,
                    facecolor="none", edgecolor=WONG["orange"], linewidths=1.1,
                    zorder=4, label="degenerate (chance)")
    axL.set_xlim(lo * 100, hi * 100)
    axL.set_ylim(lo * 100, hi * 100)
    axL.set_aspect("equal", adjustable="box")
    axL.set_xlabel("FP32 test accuracy [%]")
    axL.set_ylabel("INT8 (PTQ) test accuracy [%]")
    axL.set_title("(a)  Quantization preserves accuracy", loc="left", fontsize=9.5)
    axL.legend(loc="upper left", fontsize=7.8)

    # ---- Panel B: per-architecture accuracy change from quantization ---------
    # (Non-confounded companion to (a): the FP32 -> INT8 accuracy shift itself.
    #  We deliberately do NOT plot tabulated-NATS vs INT8 here, because that
    #  comparison is dominated by the 200- vs 12-epoch training-budget gap, not
    #  by quantization; it is reported in the text with its confound decomposed.)
    delta = (int8 - fp32) * 100.0  # pp; negative = INT8 slightly below FP32
    bins = np.arange(-2.0, 0.61, 0.2)
    axR.hist(delta, bins=bins, color=WONG["green"], alpha=0.8,
             edgecolor="white", linewidth=0.4)
    axR.axvline(0, color=WONG["gray"], lw=1.0, ls="--")
    axR.set_xlabel("INT8 $-$ FP32 accuracy change [pp]")
    axR.set_ylabel("# architectures")
    axR.set_title("(b)  Quantization barely moves accuracy", loc="left", fontsize=9.5)
    # (Numeric summary intentionally omitted here; it is stated in the figure caption.)

    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf")
    if also_png:
        fig.savefig(f"{stem}.png")
    plt.close(fig)
    print(f"  wrote {stem}.pdf" + (f" + {stem}.png" if also_png else ""))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="export_corrected/int8_rank_correlation.csv")
    ap.add_argument("--out-fig", default="figures_corrected/fig10_int8_rank")
    ap.add_argument("--out-csv", default="export_corrected/int8_rank_summary.csv")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(REPO / args.csv)))
    idx = np.array([int(r["arch_idx"]) for r in rows])
    nats = np.array([float(r["nats_cifar10"]) for r in rows])       # 0..100
    fp32 = np.array([float(r["fp32_acc"]) for r in rows])           # 0..1
    int8 = np.array([float(r["int8_acc"]) for r in rows])           # 0..1
    delta_pp = (fp32 - int8) * 100.0

    degen = fp32 <= CHANCE
    crater = (~degen) & ((fp32 - int8) > CRATER_DROP)

    def block(name, x, y, mask=None):
        if mask is not None:
            x, ii, y = x[mask], idx[mask], y[mask]
        else:
            ii = idx
        rho = spearman(x, y)
        tau = kendall_tau_b(x, y)
        o10 = topk_overlap(ii, x, y, 10)
        o25 = topk_overlap(ii, x, y, 25)
        print(f"  {name:<34s} rho={rho:.4f}  tau={tau:.4f}  "
              f"top10={o10[0]}/{o10[1]}  top25={o25[0]}/{o25[1]}  (n={len(x)})")
        return dict(comparison=name, n=len(x), spearman_rho=round(rho, 4),
                    kendall_tau=round(tau, 4),
                    top10_overlap=o10[0], top25_overlap=o25[0])

    print(f"\nINT8 rank-correlation analysis  ({len(rows)} architectures)")
    print(f"  degenerate (FP32 <= {CHANCE:.2f}, at chance): {int(degen.sum())}"
          f"  -> {sorted(idx[degen].tolist())}")
    print(f"  genuine quantization craters (FP32 above chance, INT8 drop >"
          f"{int(CRATER_DROP*100)}pp): {int(crater.sum())}")
    print(f"  |delta acc| pp : mean={np.abs(delta_pp).mean():.2f}  "
          f"median={np.median(np.abs(delta_pp)):.2f}  "
          f"max={np.abs(delta_pp).max():.2f}")
    print(f"  signed delta pp: mean={delta_pp.mean():+.2f}  "
          f"median={np.median(delta_pp):+.2f}  "
          f"(FP32 higher on {int((delta_pp > 0).sum())}/{len(rows)})")

    print("\n(a) pure quantization effect  our_FP32 -> our_INT8")
    a_all = block("all archs", fp32, int8)
    a_nd = block("excl. degenerate", fp32, int8, mask=~degen)
    print("(b) paper-relevant chain      NATS_FP32 -> our_INT8")
    b_all = block("all archs", nats, int8)
    b_nd = block("excl. degenerate", nats, int8, mask=~degen)

    # -- robustness of (a): bootstrap CI, top-band, decision regret -------------
    nd = ~degen
    print("\nRobustness of (a)  (non-degenerate, n=%d)" % int(nd.sum()))
    rho_ci = bootstrap_ci(fp32[nd], int8[nd], spearman)
    tau_ci = bootstrap_ci(fp32[nd], int8[nd], kendall_tau_b)
    print(f"  bootstrap 95% CI   rho [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}]  "
          f"tau [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}]")
    for thr in (0.75, 0.80):
        m = fp32 >= thr
        if m.sum() >= 6:
            print(f"  band FP32>={thr:.2f} (n={int(m.sum()):3d})  "
                  f"rho={spearman(fp32[m], int8[m]):.4f}  tau={kendall_tau_b(fp32[m], int8[m]):.4f}")
    order = fp32.argsort()[::-1]
    for k in (10, 25):
        tk = order[:k]
        print(f"  within top-{k} by FP32       "
              f"rho={spearman(fp32[tk], int8[tk]):.4f}  tau={kendall_tau_b(fp32[tk], int8[tk]):.4f}")
    oracle_int8 = int8.max()
    for k in (1, 3, 5):
        best_pick = int8[order[:k]].max()
        print(f"  selection regret top-{k}     "
              f"{(oracle_int8 - best_pick) * 100:+.2f} pp "
              f"(best INT8 among FP32-top{k} = {best_pick*100:.2f}%)")
    # -- decompose (b): is its gap quantization or training budget? -------------
    print("Decompose (b)  (does the gap come from quantization or training budget?)")
    print(f"  NATS_FP32 -> our_FP32 (budget only, NO quant)  rho={spearman(nats[nd], fp32[nd]):.4f}")
    print(f"  NATS_FP32 -> our_INT8 (budget + quantization)  rho={spearman(nats[nd], int8[nd]):.4f}")

    out_csv = REPO / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(a_all.keys()))
        w.writeheader()
        for tag, d in [("a_fp32_int8_all", a_all), ("a_fp32_int8_nondegen", a_nd),
                       ("b_nats_int8_all", b_all), ("b_nats_int8_nondegen", b_nd)]:
            d = dict(d); d["comparison"] = tag
            w.writerow(d)
    print(f"\n  wrote {out_csv}")

    make_figure(idx, nats, fp32, int8, degen, REPO / args.out_fig, not args.no_png)


if __name__ == "__main__":
    main()
