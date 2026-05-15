"""Reproduce all figures from the NATS-Bench HW evaluation using the export/ CSVs.

Requires only: matplotlib, numpy  (no NATS-Bench API, no server access).
Accuracy data (fig5) is read from accuracies.csv if present; omitted otherwise.
Memory figures (fig7, fig8) require the memory columns in hw_metrics.csv.

Usage:
    python plot_from_export.py                        # reads export/, writes figures_export/
    python plot_from_export.py --data export/ --out my_figures/
    python plot_from_export.py --no-png               # PDF only
"""
from __future__ import annotations

import argparse
import csv
import json
import math

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

WONG = {
    "blue":   "#0072B2",
    "orange": "#D55E00",
    "green":  "#009E73",
    "pink":   "#CC79A7",
    "yellow": "#E69F00",
    "sky":    "#56B4E9",
    "black":  "#222222",
    "gray":   "#9A9A9A",
}
DATASET_COLOR = {
    "cifar10": WONG["blue"],
    "cifar100": WONG["orange"],
    "ImageNet16-120": WONG["green"],
}
DATASET_LABEL = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "ImageNet16-120": "ImageNet-16-120",
}
NATS_BENCH_TOTAL = 15625


def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.labelpad": 3,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_dir: Path, stem: str, also_png: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    if also_png:
        fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  wrote {stem}.pdf" + (f" + {stem}.png" if also_png else ""))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _float(v):
    try:
        return float(v) if v not in ("", None) else None
    except (ValueError, TypeError):
        return None


def _int(v):
    try:
        return int(v) if v not in ("", None) else None
    except (ValueError, TypeError):
        return None


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_hw(data_dir: Path) -> list[dict]:
    rows = load_csv(data_dir / "hw_metrics.csv")
    numeric = ["int8_kb", "rom_kb", "ram_kb", "latency_s", "current_uA",
               "power_mW", "energy_mJ", "g_model_kb", "arena_used_kb",
               "arena_size_kb", "arena_used_pct", "operators"]
    result = []
    for r in rows:
        d = {"idx": int(r["arch_idx"]), "arch_str": r.get("arch_str", ""), "board": r.get("board", "")}
        for col in numeric:
            d[col] = _float(r.get(col))
        # Parse JSON category dicts (values already in KB)
        for col in ("rom_categories_json", "ram_categories_json"):
            raw = r.get(col, "")
            d[col[:-5]] = json.loads(raw) if raw else None  # strip "_json" suffix
        result.append(d)
    return result


def load_failures(data_dir: Path) -> list[dict]:
    rows = load_csv(data_dir / "failures.csv")
    result = []
    for r in rows:
        result.append({
            "idx": int(r["arch_idx"]),
            "stage": r["stage"],
            "int8_kb": _float(r.get("int8_kb")),
            "overflow_kb": _float(r.get("overflow_kb")),
            "board": r.get("board", ""),
        })
    return result


def load_coverage(data_dir: Path) -> list[dict]:
    rows = load_csv(data_dir / "coverage.csv")
    return [{"stage": r["stage"], "count": int(r["count"]),
             "pct": float(r["pct_of_total"])} for r in rows]


def load_accuracies(data_dir: Path) -> dict[str, dict[int, float]] | None:
    p = data_dir / "accuracies.csv"
    if not p.exists():
        return None
    rows = load_csv(p)
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    accs: dict[str, dict[int, float]] = {ds: {} for ds in datasets}
    for r in rows:
        idx = int(r["arch_idx"])
        for ds in datasets:
            v = _float(r.get(ds))
            if v is not None:
                accs[ds][idx] = v
    return accs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spearman(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ox = xs.argsort(); rx = np.empty_like(ox, float); rx[ox] = np.arange(len(xs))
    oy = ys.argsort(); ry = np.empty_like(oy, float); ry[oy] = np.arange(len(ys))
    return float(np.corrcoef(rx, ry)[0, 1])


def linfit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.size < 2:
        return None
    m, b = np.polyfit(xs, ys, 1)
    yhat = m * xs + b
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(m), float(b), r2


def pareto_front(costs, accs):
    order = np.argsort(costs)
    front, best = [], -np.inf
    for i in order:
        if accs[i] > best:
            front.append(i)
            best = accs[i]
    return np.array(front)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_deployment_yield(coverage, failures, out_dir, also_png):
    """Pie chart: successful vs failed vs not-evaluated across the search space."""
    n_success = next((r["count"] for r in coverage if r["stage"] == "Successful"), 0)
    n_fail = len(failures)
    n_other = NATS_BENCH_TOTAL - n_success - n_fail

    sizes  = [n_success, n_fail, n_other]
    colors = [WONG["green"], WONG["orange"], WONG["gray"]]
    labels = [
        f"Successful\n{n_success:,}  ({100*n_success/NATS_BENCH_TOTAL:.1f}%)",
        f"Failed (flash overflow)\n{n_fail:,}  ({100*n_fail/NATS_BENCH_TOTAL:.1f}%)",
        f"Not evaluated\n{n_other:,}  ({100*n_other/NATS_BENCH_TOTAL:.1f}%)",
    ]

    # Drop zero-size slices
    sizes, colors, labels = zip(*[(s, c, l) for s, c, l in zip(sizes, colors, labels) if s > 0])

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(edgecolor="white", linewidth=1.2))
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.92, 0.5),
              fontsize=8.5, frameon=False)
    ax.set_title(f"Deployment yield\n({NATS_BENCH_TOTAL:,} NATS-Bench TSS architectures)",
                 loc="center", pad=6)
    fig.tight_layout()
    save(fig, out_dir, "fig1_deployment_yield", also_png)


def fig_metric_dists(hw, out_dir, also_png):
    metrics = [
        ("rom_kb",     "ROM usage [KB]",           WONG["blue"]),
        ("latency_s",  "Inference latency [s]",    WONG["orange"]),
        ("power_mW",   "Mean power [mW]",           WONG["green"]),
        ("energy_mJ",  "Energy / inference [mJ]",  WONG["yellow"]),
        ("int8_kb",    "INT8 model size [KB]",      WONG["sky"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for ax, (k, label, c) in zip(axes.flat, metrics):
        vals = np.array([r[k] for r in hw if r[k] is not None])
        if vals.size == 0:
            ax.set_visible(False); continue
        ax.hist(vals, bins=40, color=c, alpha=0.85, edgecolor="white", linewidth=0.4)
        med, mn = float(np.median(vals)), float(np.mean(vals))
        ax.axvline(med, color=WONG["black"], lw=0.8, ls="--", alpha=0.7)
        ax.axvline(mn,  color=WONG["black"], lw=0.8, ls=":",  alpha=0.7)
        ax.text(0.97, 0.95, f"N={vals.size:,}\nmedian={med:.2f}\nmean={mn:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
        ax.set_xlabel(label)
        ax.set_ylabel("# architectures")
        ax.grid(axis="y")
    axes.flat[-1].set_visible(False)  # hide unused 6th cell
    fig.suptitle(f"Hardware metric distributions across {len(hw):,} deployed architectures",
                 fontsize=10.5, y=1.005)
    fig.tight_layout()
    save(fig, out_dir, "fig2_metric_dists", also_png)



def fig_metric_scatter(hw, out_dir, also_png):
    pairs = [
        ("rom_kb",    "ROM [KB]",    "latency_s", "Latency [s]"),
        ("rom_kb",    "ROM [KB]",    "energy_mJ", "Energy [mJ]"),
        ("latency_s", "Latency [s]", "energy_mJ", "Energy [mJ]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    for ax, (xk, xlab, yk, ylab) in zip(axes, pairs):
        xs = np.array([r[xk] for r in hw if r[xk] is not None and r[yk] is not None])
        ys = np.array([r[yk] for r in hw if r[xk] is not None and r[yk] is not None])
        ax.scatter(xs, ys, s=3.5, color=WONG["blue"], alpha=0.35, edgecolor="none")
        rho = spearman(xs, ys)
        ax.text(0.04, 0.96, f"Spearman ρ = {rho:+.3f}\nN = {len(xs):,}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.grid(True)
    fig.suptitle("Pairwise relationships between hardware metrics", fontsize=10.5, y=1.005)
    fig.tight_layout()
    save(fig, out_dir, "fig4_metric_scatter", also_png)


def fig_accuracy_pareto(hw, accuracies, out_dir, also_png):
    cost_cols = [
        ("latency_s", "Latency [s]"),
        ("energy_mJ", "Energy [mJ]"),
        ("rom_kb",    "ROM [KB]"),
    ]
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.8), sharex="col", sharey="row")
    for r_i, ds in enumerate(datasets):
        accs_dict = accuracies.get(ds, {})
        rows = [r for r in hw if r["idx"] in accs_dict and
                all(r[ck] is not None for ck, _ in cost_cols)]
        missing = len(hw) - len(rows)
        if missing:
            print(f"  [fig5] {ds}: {missing} architectures missing from accuracies.csv")
        accs = np.array([accs_dict[r["idx"]] for r in rows])
        ymin = float(accs.min()) - 3.0
        ymax = float(accs.max()) + 2.0
        for c_i, (ck, clabel) in enumerate(cost_cols):
            ax = axes[r_i, c_i]
            costs = np.array([r[ck] for r in rows])
            ax.scatter(costs, accs, s=4, color=WONG["gray"], alpha=0.4, edgecolor="none",
                       label=f"all ({len(rows):,})", rasterized=True)
            front = pareto_front(costs, accs)
            ax.plot(costs[front], accs[front],
                    color=DATASET_COLOR[ds], lw=1.2, marker="o",
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                    label=f"Pareto front ({len(front)})")
            ax.set_ylim(ymin, ymax)
            ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= ymax])
            if c_i == 0:
                ax.set_ylabel(f"{DATASET_LABEL[ds]}\ntest accuracy [%]")
            if r_i == 2:
                ax.set_xlabel(clabel)
            if r_i == 0:
                ax.set_title(clabel.split(" [")[0], loc="left", pad=3, fontsize=10.5)
            ax.grid(True)
            ax.legend(loc="lower right", framealpha=0.9, edgecolor="none", facecolor="white")
    fig.suptitle("Accuracy vs. hardware cost — Pareto frontiers per dataset",
                 fontsize=11, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, out_dir, "fig5_accuracy_pareto", also_png)


def fig_per_board_boxplots(hw, out_dir, also_png):
    metrics = [
        ("power_mW",   "Mean power [mW]"),
        ("current_uA", "Mean current [µA]"),
        ("latency_s",  "Latency [s]"),
    ]
    boards = sorted({r["board"] for r in hw if r["board"]})
    if not boards:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for ax, (k, label) in zip(axes, metrics):
        data = [[r[k] for r in hw if r["board"] == b and r[k] is not None] for b in boards]
        bp_kwargs = dict(
            showfliers=True, widths=0.55, patch_artist=True,
            medianprops=dict(color=WONG["black"], lw=1.0),
            flierprops=dict(marker="o", markersize=1.8, alpha=0.25,
                            markerfacecolor=WONG["gray"], markeredgecolor="none"),
        )
        try:
            bp = ax.boxplot(data, tick_labels=boards, **bp_kwargs)
        except TypeError:
            bp = ax.boxplot(data, labels=boards, **bp_kwargs)
        palette = [WONG["blue"], WONG["orange"], WONG["green"], WONG["pink"]]
        for patch, c in zip(bp["boxes"], palette):
            patch.set_facecolor(c); patch.set_alpha(0.7); patch.set_edgecolor("white")
        for whisk in bp["whiskers"] + bp["caps"]:
            whisk.set_color("#666666"); whisk.set_linewidth(0.7)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y")
    fig.suptitle("Per-board distribution check — no systematic bias", fontsize=10.5, y=1.005)
    fig.tight_layout()
    save(fig, out_dir, "fig6_per_board_boxplots", also_png)


def fig_memory_composition(hw, out_dir, also_png):
    # Same top-5 + relabel logic as plot_results.py
    def relabel(cat):
        return {"(no paths)": "Application", "ZEPHYR_BASE": "Zephyr OS",
                "WORKSPACE": "TFLM runtime", "(hidden)": "Linker (hidden)",
                "OUTPUT_DIR": "Output dir", "/": "Toolchain"}.get(cat, cat)

    def ordered(d, top_k=5):
        items = sorted(d.items(), key=lambda kv: -kv[1])
        head = items[:top_k]
        other = sum(v for _, v in items[top_k:])
        if other:
            head.append(("Other", other))
        return head

    def mean_per_cat(rows, key):
        agg = {}
        for r in rows:
            cats = r.get(key) or {}
            for cat, val in cats.items():
                agg.setdefault(cat, []).append(float(val))
        return {cat: float(sum(v) / len(v)) for cat, v in agg.items()}

    with_rom = [r for r in hw if r.get("rom_categories") and r.get("g_model_kb") is not None]
    with_ram = [r for r in hw if r.get("ram_categories")]
    if not with_rom or not with_ram:
        print("  [skip] fig7: memory columns absent from hw_metrics.csv (run export without --no-memory)")
        return

    rom_means = mean_per_cat(with_rom, "rom_categories")
    ram_means = mean_per_cat(with_ram, "ram_categories")
    rom_items = ordered(rom_means)
    ram_items = ordered(ram_means)

    all_cats = []
    for cat, _ in rom_items + ram_items:
        if cat not in all_cats:
            all_cats.append(cat)

    palette = [WONG["blue"], WONG["orange"], WONG["green"], WONG["pink"],
               WONG["yellow"], WONG["sky"], WONG["gray"]]
    cat_color = {cat: palette[i % len(palette)] for i, cat in enumerate(all_cats)}

    fig, ax = plt.subplots(figsize=(6.0, 2.8))

    bar_h = 0.55
    legend_seen = set()
    for y, items in [(1, rom_items), (0, ram_items)]:
        total = sum(v for _, v in items)
        left = 0.0
        for cat, val in items:
            lbl = relabel(cat)
            handle_lbl = lbl if cat not in legend_seen else None
            legend_seen.add(cat)
            ax.barh(y, val, left=left, color=cat_color[cat], edgecolor="white",
                    height=bar_h, linewidth=0.5, label=handle_lbl)
            if total > 0 and val / total > 0.10:
                ax.text(left + val / 2, y, f"{val:,.0f} KB",
                        ha="center", va="center", fontsize=8.0, color="white", fontweight="bold")
            left += val
        ax.text(left + total * 0.012, y, f"Σ {total:,.0f} KB",
                ha="left", va="center", fontsize=8.0, color="#444444")

    ax.set_yticks([1, 0], ["ROM\n(flash)", "RAM\n(SRAM)"])
    ax.set_xlabel("Mean size [KB] across deployed architectures")
    ax.set_title("Memory composition: where the 1 MB flash budget goes", loc="left", pad=4)
    ax.grid(axis="x")
    max_total = max(sum(v for _, v in rom_items), sum(v for _, v in ram_items))
    ax.set_xlim(0, max_total * 1.18)
    ax.tick_params(left=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=4,
              fontsize=8.0, frameon=False)

    fig.tight_layout()
    save(fig, out_dir, "fig7_memory_composition", also_png)


def fig_arena_utilization(hw, out_dir, also_png):
    with_arena = [r for r in hw if r.get("arena_used_kb") is not None]
    if not with_arena:
        print("  [skip] fig8: arena_used_kb absent from hw_metrics.csv")
        return

    used_kb = np.array([r["arena_used_kb"] for r in with_arena])
    sizes = [r["arena_size_kb"] for r in with_arena if r.get("arena_size_kb") is not None]
    arena_alloc = float(sizes[0]) if sizes else 350.0
    max_used = float(used_kb.max())
    mean_used = float(used_kb.mean())

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax_a, ax_b = axes

    # Panel (a): histogram with reference markers
    bins = np.linspace(0, arena_alloc * 1.04, 50)
    ax_a.hist(used_kb, bins=bins, color=WONG["blue"], alpha=0.8,
              edgecolor="white", linewidth=0.4)
    ymax = ax_a.get_ylim()[1]
    ax_a.axvline(arena_alloc, color=WONG["orange"], lw=1.4, ls="--")
    ax_a.axvline(max_used,    color=WONG["green"],  lw=1.0, ls=":")
    ax_a.axvline(mean_used,   color=WONG["black"],  lw=0.8, ls="-", alpha=0.6)
    ax_a.text(arena_alloc - 4, ymax * 0.97, f"allocated\n{arena_alloc:.0f} KB",
              fontsize=8.0, ha="right", va="top", color=WONG["orange"])
    ax_a.text(max_used + 6, ymax * 0.97, f"max\nobserved\n{max_used:.0f} KB",
              fontsize=8.0, ha="left", va="top", color=WONG["green"])
    ax_a.text(mean_used - 4, ymax * 0.45, f"mean\n{mean_used:.0f} KB",
              fontsize=7.8, ha="right", va="top", color=WONG["black"])
    ax_a.text(0.98, 0.55,
              f"N = {len(used_kb):,}\n"
              f"mean usage = {100*mean_used/arena_alloc:.0f}% of arena\n"
              f"max usage  = {100*max_used/arena_alloc:.0f}% of arena",
              transform=ax_a.transAxes, va="top", ha="right", fontsize=8.0,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5))
    ax_a.set_xlabel("Tensor arena used [KB]")
    ax_a.set_ylabel("# architectures")
    ax_a.grid(axis="y")
    ax_a.set_title("(a) Arena utilization", loc="left", pad=4)

    # Panel (b): g_model vs arena_used scatter + regression
    with_both = [r for r in with_arena if r.get("g_model_kb") is not None]
    if with_both:
        xs = np.array([r["g_model_kb"] for r in with_both])
        ys = np.array([r["arena_used_kb"] for r in with_both])
        ax_b.scatter(xs, ys, s=4, color=WONG["blue"], alpha=0.4, edgecolor="none", rasterized=True)
        fit = linfit(xs, ys)
        if fit:
            m, b, r2 = fit
            xs_line = np.linspace(xs.min(), xs.max(), 100)
            ax_b.plot(xs_line, m * xs_line + b, color=WONG["orange"], lw=1.2)
            ax_b.text(0.97, 0.04,
                      f"used ≈ {m:.3f}·g_model + {b:.1f} KB\n"
                      f"R² = {r2:.3f}   N = {len(xs):,}",
                      transform=ax_b.transAxes, va="bottom", ha="right", fontsize=8.0,
                      bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.9, pad=2.5))
        ax_b.axhline(arena_alloc, color=WONG["orange"], lw=1.0, ls="--", alpha=0.8)
        ax_b.text(xs.min(), arena_alloc - 4, f"allocated arena ({arena_alloc:.0f} KB)",
                  ha="left", va="top", fontsize=7.8, color=WONG["orange"])
        ax_b.set_ylim(0, arena_alloc * 1.05)
    ax_b.set_xlabel("g_model symbol [KB]")
    ax_b.set_ylabel("Arena used at runtime [KB]")
    ax_b.set_title("(b) Runtime arena usage vs. model size (R²=weak: size alone cannot predict usage)", loc="left", pad=4)
    ax_b.grid(True)

    fig.suptitle("Tensor arena: static allocation covers worst-case runtime usage across all architectures",
                 fontsize=10.5, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, out_dir, "fig8_arena_utilization", also_png)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="export", type=Path,
                   help="Directory containing the CSV files (default: export/)")
    p.add_argument("--out", default="figures_export", type=Path,
                   help="Output directory for figures (default: figures_export/)")
    p.add_argument("--no-png", action="store_true", help="PDF only, skip PNG output.")
    args = p.parse_args()

    set_pub_style()
    also_png = not args.no_png

    print(f"Data dir : {args.data}")
    print(f"Out dir  : {args.out}")

    print("Loading CSVs...")
    hw       = load_hw(args.data)
    failures = load_failures(args.data)
    coverage = load_coverage(args.data)
    accs     = load_accuracies(args.data)
    print(f"  {len(hw):,} successful  |  {len(failures):,} failed  |  "
          f"accuracies: {'yes' if accs else 'no (accuracies.csv not found)'}")

    print("\nRendering figures:")
    fig_deployment_yield(coverage, failures, args.out, also_png)
    fig_metric_dists(hw, args.out, also_png)
    fig_metric_scatter(hw, args.out, also_png)
    fig_per_board_boxplots(hw, args.out, also_png)
    fig_memory_composition(hw, args.out, also_png)
    fig_arena_utilization(hw, args.out, also_png)

    if accs:
        fig_accuracy_pareto(hw, accs, args.out, also_png)
    else:
        print("  [skipped fig5_accuracy_pareto — accuracies.csv not found]")

    print(f"\nDone. Figures in {args.out}/")


if __name__ == "__main__":
    main()
