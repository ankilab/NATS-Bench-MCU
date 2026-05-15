"""Generate publication-quality figures for the NATS-Bench HW evaluation.

Reuses the data-loading code from `analyze_artifacts.py`. Produces PDF + PNG
under `figures/`. Default source is the server folder tree (largest dataset);
use `--source local` to read from hw-nats-bench/artifacts/ instead.

Figures produced:
    fig1_coverage_failures  -- pipeline funnel + failure stage breakdown
    fig2_metric_dists       -- 2x3 grid of HW-metric histograms
    fig3_flash_threshold    -- INT8 size of failed vs successful, with precheck cutoff
    fig4_metric_scatter     -- pairwise HW-metric scatter (2x2 grid)
    fig5_accuracy_pareto    -- 3x3 accuracy vs cost grid with Pareto fronts (needs NATS API)
    fig6_per_board_boxplots -- power/current/latency by board (bias check)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import analyze_artifacts as aa


# --------- publication style ---------

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
DATASET_COLOR = {"cifar10": WONG["blue"], "cifar100": WONG["orange"], "ImageNet16-120": WONG["green"]}
DATASET_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "ImageNet16-120": "ImageNet-16-120"}


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
        "pdf.fonttype": 42,  # editable text in PDFs
        "ps.fonttype": 42,
    })


def save(fig, out_dir: Path, stem: str, also_png: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    if also_png:
        fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  wrote {stem}.pdf" + (f" + {stem}.png" if also_png else ""))


# --------- data preparation ---------

def gather_successes(records, manifest):
    """Return numpy struct of all successful runs ready to plot."""
    rows = []
    for idx, r in records.items():
        res = r.get("results") or {}
        if r["meta_status"] != "done" or not res:
            continue
        m = manifest.get(idx, {})
        uart = r.get("uart") or {}
        ram = r.get("ram") or {}
        rom = r.get("rom") or {}
        row = {
            "idx": idx,
            "arch_str": m.get("arch_str", ""),
            "int8_kb": (m.get("int8_size_bytes") or 0) / 1024.0,
            "rom_kb": res.get("rom_usage", 0) / 1024.0,
            "ram_kb": res.get("ram_usage", 0) / 1024.0,
            "latency_s": res.get("duration_avg_s"),
            "current_uA": res.get("avg_current_uA"),
            "power_mW": (res.get("avg_power_uW") or 0) / 1000.0,
            "energy_mJ": (res.get("avg_energy_uJ") or 0) / 1000.0,
            "board": r.get("board", ""),
            # memory-layout fields (None if --with-memory wasn't passed during scan)
            "arena_size_kb": (uart.get("arena_size") or 0) / 1024.0 or None,
            "arena_used_kb": (uart.get("arena_used") or 0) / 1024.0 or None,
            "arena_used_pct": uart.get("arena_used_pct"),
            "operators": uart.get("operators"),
            "ram_categories": ram.get("by_category") if ram else None,
            "rom_categories": rom.get("by_category") if rom else None,
            "g_model_kb": ((rom.get("key_symbols") or {}).get("g_model") or 0) / 1024.0 if rom else None,
        }
        rows.append(row)
    return rows


def gather_failures(records, manifest):
    fails = []
    for idx, r in records.items():
        if r["meta_status"] != "error":
            continue
        m = manifest.get(idx, {})
        stage = r["fail_stage"]
        if stage == "flash_failed":
            stage = "precheck"
        fails.append({
            "idx": idx,
            "stage": stage,
            "int8_kb": (m.get("int8_size_bytes") or 0) / 1024.0,
            "overflow_kb": r.get("overflow_kb"),
            "board": r.get("board", ""),
        })
    return fails


# --------- figures ---------

def fig_coverage_failures(records, submitted_idx, manifest, out_dir, also_png):
    n_manifest = sum(1 for r in manifest.values() if r["status"] == "ok")
    n_sub = len(submitted_idx)
    n_present = len(records)
    n_done = sum(1 for r in records.values() if r["meta_status"] == "done")
    n_error = sum(1 for r in records.values() if r["meta_status"] == "error")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Left: pipeline funnel as horizontal bars
    ax = axes[0]
    stages = ["NATS-Bench\n(target)", "TFLite\nconverted", "Submitted", "On disk", "Successful"]
    counts = [aa.NATS_BENCH_TOTAL, n_manifest, n_sub, n_present, n_done]
    colors = [WONG["gray"], WONG["sky"], WONG["blue"], WONG["green"], WONG["orange"]]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, counts, color=colors, edgecolor="white", height=0.7)
    for yi, c in zip(y, counts):
        ax.text(c + aa.NATS_BENCH_TOTAL * 0.012, yi,
                f"{c:,}  ({100*c/aa.NATS_BENCH_TOTAL:.1f}%)",
                va="center", fontsize=8.5)
    ax.set_yticks(y, stages)
    ax.set_xlim(0, aa.NATS_BENCH_TOTAL * 1.22)
    ax.set_xlabel("Number of architectures")
    ax.set_title("(a) Pipeline coverage", loc="left", pad=4)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x else "0"))
    ax.grid(axis="x")
    ax.tick_params(left=False)

    # Right: failure stage breakdown
    ax = axes[1]
    from collections import Counter
    stage_counts = Counter(
        ("precheck" if r["fail_stage"] == "flash_failed" else r["fail_stage"])
        for r in records.values() if r["meta_status"] == "error"
    )
    if stage_counts:
        labels, vals = zip(*stage_counts.most_common())
        ys = np.arange(len(labels))[::-1]
        bar_colors = [WONG["pink"], WONG["yellow"], WONG["sky"], WONG["gray"]][:len(labels)]
        ax.barh(ys, vals, color=bar_colors, edgecolor="white", height=0.7)
        for yi, v in zip(ys, vals):
            ax.text(v + max(vals) * 0.02, yi,
                    f"{v:,} ({100*v/n_error:.1f}%)", va="center", fontsize=8.5)
        ax.set_yticks(ys, labels)
        ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel(f"Failed evaluations (total = {n_error:,})")
    ax.set_title("(b) Failures by stage", loc="left", pad=4)
    ax.grid(axis="x")
    ax.tick_params(left=False)

    fig.tight_layout()
    save(fig, out_dir, "fig1_coverage_failures", also_png)


def fig_metric_dists(succ, out_dir, also_png):
    metrics = [
        ("rom_kb",     "ROM usage [KB]",       WONG["blue"]),
        ("latency_s",  "Inference latency [s]", WONG["orange"]),
        ("current_uA", "Mean current [µA]",     WONG["green"]),
        ("power_mW",   "Mean power [mW]",       WONG["pink"]),
        ("energy_mJ",  "Energy / inference [mJ]", WONG["yellow"]),
        ("int8_kb",    "INT8 model size [KB]",  WONG["sky"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for ax, (k, label, c) in zip(axes.flat, metrics):
        vals = np.array([r[k] for r in succ if r.get(k) is not None])
        if vals.size == 0:
            ax.set_visible(False); continue
        ax.hist(vals, bins=40, color=c, alpha=0.85, edgecolor="white", linewidth=0.4)
        med = float(np.median(vals)); mn = float(np.mean(vals))
        ax.axvline(med, color=WONG["black"], lw=0.8, ls="--", alpha=0.7)
        ax.axvline(mn, color=WONG["black"], lw=0.8, ls=":", alpha=0.7)
        ax.text(0.97, 0.95, f"N={vals.size:,}\nmedian={med:.2f}\nmean={mn:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
        ax.set_xlabel(label)
        ax.set_ylabel("# architectures")
        ax.grid(axis="y")
    fig.suptitle(f"Hardware metric distributions across {len(succ):,} deployed architectures",
                 fontsize=10.5, y=1.005)
    fig.tight_layout()
    save(fig, out_dir, "fig2_metric_dists", also_png)


def fig_flash_threshold(succ, fails, out_dir, also_png):
    succ_kb = np.array([r["int8_kb"] for r in succ if r.get("int8_kb")])
    fail_kb = np.array([r["int8_kb"] for r in fails if r.get("int8_kb")])
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bins = np.linspace(0, max(succ_kb.max(), fail_kb.max()) * 1.02, 60)
    ax.hist(succ_kb, bins=bins, color=WONG["green"], alpha=0.7, label=f"Successful ({succ_kb.size:,})",
            edgecolor="white", linewidth=0.3)
    ax.hist(fail_kb, bins=bins, color=WONG["orange"], alpha=0.75, label=f"Failed ({fail_kb.size:,})",
            edgecolor="white", linewidth=0.3)
    ax.axvline(800, color=WONG["black"], lw=1.0, ls="--", zorder=0)
    ymax = ax.get_ylim()[1]
    ax.annotate("precheck threshold\n(80% of 1 MB flash)",
                xy=(800, ymax * 0.30), xytext=(1180, ymax * 0.55),
                fontsize=8.0, color=WONG["black"], ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=WONG["black"], lw=0.6,
                                connectionstyle="arc3,rad=-0.15"))
    ax.legend(loc="upper right")
    ax.set_xlabel("INT8 model size [KB]")
    ax.set_ylabel("# architectures")
    ax.set_title("INT8 model size vs. deployment outcome", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, out_dir, "fig3_flash_threshold", also_png)


def fig_metric_scatter(succ, out_dir, also_png):
    pairs = [
        ("rom_kb",    "ROM [KB]",       "latency_s", "Latency [s]"),
        ("rom_kb",    "ROM [KB]",       "energy_mJ", "Energy [mJ]"),
        ("latency_s", "Latency [s]",    "energy_mJ", "Energy [mJ]"),
        ("int8_kb",   "INT8 size [KB]", "rom_kb",    "ROM [KB]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0))
    for ax, (xk, xlab, yk, ylab) in zip(axes.flat, pairs):
        xs = np.array([r[xk] for r in succ])
        ys = np.array([r[yk] for r in succ])
        ax.scatter(xs, ys, s=3.5, color=WONG["blue"], alpha=0.35, edgecolor="none")
        # Spearman rank correlation
        order_x = xs.argsort(); rx = np.empty_like(order_x, dtype=float); rx[order_x] = np.arange(len(xs))
        order_y = ys.argsort(); ry = np.empty_like(order_y, dtype=float); ry[order_y] = np.arange(len(ys))
        rho = np.corrcoef(rx, ry)[0, 1]
        ax.text(0.04, 0.96, f"Spearman ρ = {rho:+.3f}\nN = {len(xs):,}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.grid(True)
    fig.suptitle("Pairwise relationships between hardware metrics", fontsize=10.5, y=1.005)
    fig.tight_layout()
    save(fig, out_dir, "fig4_metric_scatter", also_png)


def pareto_indices(costs, accs):
    """Return indices on the Pareto front (minimize cost, maximize accuracy)."""
    order = np.argsort(costs)  # ascending cost
    front = []
    best_acc = -np.inf
    for i in order:
        if accs[i] > best_acc:
            front.append(i)
            best_acc = accs[i]
    return np.array(front)


def fig_accuracy_pareto(succ, accuracies, out_dir, also_png):
    """3x3 grid: rows = datasets, cols = cost metrics.

    Y-axis spans the full observed accuracy range for each dataset (no
    clipping) with ~3 pp of whitespace below the lowest point. The very-low
    cluster (mostly all-`none` cells) is the true bottom of NATS-Bench TSS.
    """
    cost_cols = [
        ("latency_s", "Latency [s]"),
        ("energy_mJ", "Energy [mJ]"),
        ("rom_kb",    "ROM [KB]"),
    ]
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    bottom_margin = 3.0  # percentage points of whitespace below lowest data
    top_margin = 2.0

    fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.8), sharex="col", sharey="row")
    for r_i, ds in enumerate(datasets):
        accs_dict = accuracies.get(ds, {})
        rows = [r for r in succ if r["idx"] in accs_dict]
        missing = len(succ) - len(rows)
        if missing:
            print(f"  [fig5] {ds}: {missing} of {len(succ):,} architectures missing "
                  f"from accuracy cache (showing the remaining {len(rows):,}).")
        accs = np.array([accs_dict[r["idx"]] for r in rows])
        n_total = len(accs)
        ymin = float(accs.min()) - bottom_margin
        ymax = float(accs.max()) + top_margin
        for c_i, (ck, clabel) in enumerate(cost_cols):
            ax = axes[r_i, c_i]
            costs = np.array([r[ck] for r in rows])
            ax.scatter(costs, accs, s=4, color=WONG["gray"], alpha=0.4, edgecolor="none",
                       label=f"all ({n_total:,})", rasterized=True)
            front = pareto_indices(costs, accs)
            ax.plot(costs[front], accs[front], color=DATASET_COLOR[ds], lw=1.2, marker="o",
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                    label=f"Pareto front ({len(front)})")
            ax.set_ylim(ymin, ymax)
            # Keep tick labels non-negative — the margin below 0 stays as whitespace.
            ax.set_yticks([t for t in ax.get_yticks() if t >= 0 and t <= ymax])
            if c_i == 0:
                ax.set_ylabel(f"{DATASET_LABEL[ds]}\ntest accuracy [%]")
            if r_i == 2:
                ax.set_xlabel(clabel)
            if r_i == 0:
                ax.set_title(clabel.split(" [")[0], loc="left", pad=3, fontsize=10.5)
            ax.grid(True)
            ax.legend(loc="lower right", framealpha=0.9, edgecolor="none",
                      facecolor="white")
    fig.suptitle("Accuracy vs. hardware cost — Pareto frontiers per dataset",
                 fontsize=11, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, out_dir, "fig5_accuracy_pareto", also_png)


def fig_per_board_boxplots(succ, out_dir, also_png):
    metrics = [
        ("power_mW",   "Mean power [mW]"),
        ("current_uA", "Mean current [µA]"),
        ("latency_s",  "Latency [s]"),
    ]
    boards = sorted({r["board"] for r in succ if r.get("board")})
    if not boards:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for ax, (k, label) in zip(axes, metrics):
        data = [[r[k] for r in succ if r["board"] == b and r.get(k) is not None] for b in boards]
        bp_kwargs = dict(
            showfliers=True, widths=0.55, patch_artist=True,
            medianprops=dict(color=WONG["black"], lw=1.0),
            flierprops=dict(marker="o", markersize=1.8, alpha=0.25,
                            markerfacecolor=WONG["gray"], markeredgecolor="none"),
        )
        # Matplotlib 3.9 renamed `labels` -> `tick_labels`.
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


def _linfit(xs, ys):
    """Return (slope, intercept, R^2) for the best-fit line y = mx + b."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return None
    m, b = np.polyfit(xs, ys, 1)
    yhat = m * xs + b
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(m), float(b), r2


def fig_memory_composition(succ, out_dir, also_png):
    """1x2: (a) mean RAM/ROM composition stacked bars; (b) g_model vs ROM total."""
    with_rom = [r for r in succ if r.get("rom_categories")]
    with_ram = [r for r in succ if r.get("ram_categories")]
    if not with_rom or not with_ram:
        print("  [skip] fig7_memory_composition: no ram/rom data "
              "(run with memory parsing enabled).")
        return

    # Aggregate top categories
    def mean_per_cat(rs, key):
        agg = {}
        for r in rs:
            for cat, sz in r[key].items():
                agg.setdefault(cat, []).append(sz / 1024.0)
        return {cat: float(np.mean(v)) for cat, v in agg.items()}

    ram_means = mean_per_cat(with_ram, "ram_categories")
    rom_means = mean_per_cat(with_rom, "rom_categories")

    # Pretty names
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

    fig = plt.figure(figsize=(9.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    palette = [WONG["blue"], WONG["orange"], WONG["green"], WONG["pink"],
               WONG["yellow"], WONG["sky"], WONG["gray"]]

    # Panel (a): stacked horizontal bars - one for RAM, one for ROM.
    # Use a unified category color map so legend covers both bars.
    rom_items = ordered(rom_means)
    ram_items = ordered(ram_means)
    all_cats = []
    for cat, _ in rom_items + ram_items:
        if cat not in all_cats:
            all_cats.append(cat)
    cat_color = {cat: palette[i % len(palette)] for i, cat in enumerate(all_cats)}

    bar_h = 0.55
    y_positions = [1, 0]
    legend_seen = set()
    for y, items, label_total in (
        (y_positions[0], rom_items, sum(v for _, v in rom_items)),
        (y_positions[1], ram_items, sum(v for _, v in ram_items)),
    ):
        left = 0.0
        for cat, val in items:
            lbl = relabel(cat)
            handle_lbl = lbl if cat not in legend_seen else None
            legend_seen.add(cat)
            ax_a.barh(y, val, left=left, color=cat_color[cat],
                      edgecolor="white", height=bar_h, linewidth=0.5,
                      label=handle_lbl)
            # In-bar size text only when the slice is wide enough
            if val / label_total > 0.10:
                ax_a.text(left + val / 2, y, f"{val:,.0f} KB",
                          ha="center", va="center", fontsize=8.0, color="white",
                          fontweight="bold")
            left += val
        ax_a.text(left + label_total * 0.012, y, f"Σ {label_total:,.0f} KB",
                  ha="left", va="center", fontsize=8.0, color="#444444")
    ax_a.set_yticks(y_positions, ["ROM\n(flash)", "RAM\n(SRAM)"])
    ax_a.set_xlabel("Mean size [KB] across deployed architectures")
    ax_a.set_title("(a) Average memory layout", loc="left", pad=4)
    ax_a.grid(axis="x")
    max_total = max(sum(v for _, v in rom_items), sum(v for _, v in ram_items))
    ax_a.set_xlim(0, max_total * 1.18)  # room for sum label on the right
    ax_a.tick_params(left=False)
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=4,
                fontsize=8.0, frameon=False)

    # Panel (b): g_model size vs total ROM, regression to recover constant overhead
    xs = np.array([r["g_model_kb"] for r in with_rom])
    ys = np.array([r["rom_kb"] for r in with_rom])
    ax_b.scatter(xs, ys, s=4, color=WONG["blue"], alpha=0.4, edgecolor="none",
                 rasterized=True)
    fit = _linfit(xs, ys)
    if fit:
        m, b, r2 = fit
        xs_line = np.linspace(xs.min(), xs.max(), 100)
        ax_b.plot(xs_line, m * xs_line + b, color=WONG["orange"], lw=1.2)
        ax_b.text(0.04, 0.96,
                  f"ROM ≈ {m:.2f}·g_model + {b:,.0f} KB\n"
                  f"R² = {r2:.3f}   N = {len(xs):,}\n"
                  f"⇒ non-model overhead ≈ {b:,.0f} KB",
                  transform=ax_b.transAxes, va="top", ha="left", fontsize=8.0,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0))
    ax_b.set_xlabel("g_model symbol [KB]")
    ax_b.set_ylabel("Total ROM [KB]")
    ax_b.set_title("(b) ROM = model + ~constant runtime", loc="left", pad=4)
    ax_b.grid(True)

    fig.suptitle("Memory composition: where the 1 MB flash budget goes",
                 fontsize=10.5, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, out_dir, "fig7_memory_composition", also_png)


def fig_arena_utilization(succ, out_dir, also_png):
    """1x2: (a) arena_used histogram with allocated/max markers;
            (b) g_model vs arena_used (predicting required arena per model)."""
    with_arena = [r for r in succ if r.get("arena_used_kb") is not None]
    if not with_arena:
        print("  [skip] fig8_arena_utilization: no uart arena data.")
        return

    used_kb = np.array([r["arena_used_kb"] for r in with_arena])
    sizes_kb = np.array([r["arena_size_kb"] for r in with_arena if r.get("arena_size_kb")])
    arena_alloc = float(sizes_kb[0]) if sizes_kb.size else 350.0
    max_used = float(used_kb.max())
    mean_used = float(used_kb.mean())
    median_used = float(np.median(used_kb))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax_a, ax_b = axes

    # Panel (a): histogram of used KB with reference markers
    bins = np.linspace(0, arena_alloc * 1.04, 50)
    ax_a.hist(used_kb, bins=bins, color=WONG["blue"], alpha=0.8,
              edgecolor="white", linewidth=0.4)
    ymax = ax_a.get_ylim()[1]
    ax_a.axvline(arena_alloc, color=WONG["orange"], lw=1.4, ls="--")
    ax_a.axvline(max_used, color=WONG["green"], lw=1.0, ls=":")
    ax_a.axvline(mean_used, color=WONG["black"], lw=0.8, ls="-", alpha=0.6)
    # Place markers high in their respective regions
    ax_a.text(arena_alloc - 4, ymax * 0.97, f"allocated\n{arena_alloc:.0f} KB",
              fontsize=8.0, ha="right", va="top", color=WONG["orange"])
    ax_a.text(max_used + 6, ymax * 0.97, f"max\nobserved\n{max_used:.0f} KB",
              fontsize=8.0, ha="left", va="top", color=WONG["green"])
    ax_a.text(mean_used - 4, ymax * 0.45, f"mean\n{mean_used:.0f} KB",
              fontsize=7.8, ha="right", va="top", color=WONG["black"])
    headroom_pct = 100 * (arena_alloc - max_used) / arena_alloc
    ax_a.text(0.98, 0.55,
              f"N = {len(used_kb):,}\n"
              f"mean usage = {100*mean_used/arena_alloc:.0f}% of arena\n"
              f"max usage  = {100*max_used/arena_alloc:.0f}% of arena\n"
              f"⇒ {headroom_pct:.0f}% headroom even for\n"
              f"   the worst-case model",
              transform=ax_a.transAxes, va="top", ha="right", fontsize=8.0,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5))
    ax_a.set_xlabel("Tensor arena used [KB]")
    ax_a.set_ylabel("# architectures")
    ax_a.grid(axis="y")
    ax_a.set_title("(a) Arena utilization", loc="left", pad=4)

    # Panel (b): g_model vs arena_used scatter + fit
    with_both = [r for r in with_arena if r.get("g_model_kb") is not None]
    if with_both:
        xs = np.array([r["g_model_kb"] for r in with_both])
        ys = np.array([r["arena_used_kb"] for r in with_both])
        ax_b.scatter(xs, ys, s=4, color=WONG["blue"], alpha=0.4, edgecolor="none",
                     rasterized=True)
        fit = _linfit(xs, ys)
        if fit:
            m, b, r2 = fit
            xs_line = np.linspace(xs.min(), xs.max(), 100)
            ax_b.plot(xs_line, m * xs_line + b, color=WONG["orange"], lw=1.2)
            # Bottom-right corner is empty (low arena usage at large model size
            # would land below the regression line, but there are few such points)
            ax_b.text(0.97, 0.04,
                      f"used ≈ {m:.3f}·g_model + {b:.1f} KB\n"
                      f"R² = {r2:.3f}   N = {len(xs):,}",
                      transform=ax_b.transAxes, va="bottom", ha="right", fontsize=8.0,
                      bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.9, pad=2.5))
        ax_b.axhline(arena_alloc, color=WONG["orange"], lw=1.0, ls="--", alpha=0.8)
        ax_b.text(xs.min(), arena_alloc - 4,
                  f"allocated arena ({arena_alloc:.0f} KB)",
                  ha="left", va="top", fontsize=7.8, color=WONG["orange"])
        ax_b.set_ylim(0, arena_alloc * 1.05)
    ax_b.set_xlabel("g_model symbol [KB]")
    ax_b.set_ylabel("Arena used at runtime [KB]")
    ax_b.set_title("(b) Arena requirement vs. model size", loc="left", pad=4)
    ax_b.grid(True)

    fig.suptitle("Tensor arena: the 350 KB static allocation is heavily over-provisioned",
                 fontsize=10.5, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, out_dir, "fig8_arena_utilization", also_png)


# --------- accuracy loading + caching ---------

def load_accuracies(succ, bench_path, cache_path, datasets):
    """Return {dataset: {idx: test_acc}}, populating cache if needed."""
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    needed = {ds: {} for ds, _ in datasets}
    missing_indices = set()
    for r in succ:
        idx_s = str(r["idx"])
        for ds, _ in datasets:
            cached = cache.get(ds, {}).get(idx_s)
            if cached is not None:
                needed[ds][r["idx"]] = float(cached)
            else:
                missing_indices.add(r["idx"])

    if not missing_indices:
        return needed

    try:
        from nats_bench import create
        api = create(str(bench_path), "tss", fast_mode=True, verbose=False)
    except Exception as e:
        print(f"  [warn] could not load NATS-Bench API ({e}); using cache only.")
        return needed

    print(f"  Querying NATS-Bench for {len(missing_indices):,} architectures "
          f"x {len(datasets)} datasets...", flush=True)
    cache.setdefault("cifar10", {}); cache.setdefault("cifar100", {}); cache.setdefault("ImageNet16-120", {})
    done = 0
    for idx in sorted(missing_indices):
        for ds, _ in datasets:
            try:
                info = api.get_more_info(idx, ds, hp="200")
                acc = float(info.get("test-accuracy"))
                cache[ds][str(idx)] = acc
                needed[ds][idx] = acc
            except Exception:
                pass
        done += 1
        if done % 1000 == 0:
            print(f"    ...{done:,}/{len(missing_indices):,}", flush=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    print(f"  cached {sum(len(v) for v in cache.values()):,} accuracy lookups -> {cache_path}")
    return needed


# --------- main ---------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["local", "server"], default="server",
                   help="Where to read run folders from (default: server).")
    p.add_argument("--artifacts", default=str(aa.DEFAULT_ARTIFACTS), type=Path)
    p.add_argument("--server-root", default=str(aa.DEFAULT_SERVER_ROOT), type=Path)
    p.add_argument("--manifest", default=str(aa.DEFAULT_MANIFEST), type=Path)
    p.add_argument("--submitted", default=str(aa.DEFAULT_SUBMITTED), type=Path)
    p.add_argument("--bench", default=str(aa.DEFAULT_BENCH), type=Path)
    p.add_argument("--out", default=str(Path(__file__).parent / "figures"), type=Path,
                   help="Output directory for figures.")
    p.add_argument("--no-png", action="store_true", help="Skip PNG output (PDF only).")
    p.add_argument("--skip-accuracy", action="store_true",
                   help="Skip fig5 (no NATS-Bench API lookups).")
    p.add_argument("--no-memory", action="store_true",
                   help="Skip fig7/fig8 and the ram/rom/uart parsing pass (faster).")
    p.add_argument("--cache", default=str(Path(__file__).parent / "figures" / ".accuracy_cache.json"),
                   type=Path, help="Accuracy lookup cache file.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    set_pub_style()

    print(f"Source     : {args.source}")
    print(f"Output dir : {args.out}")
    manifest = aa.load_manifest(args.manifest)
    submitted_idx = aa.load_submitted(args.submitted)
    folders = aa.discover_local(args.artifacts) if args.source == "local" \
              else aa.discover_server(args.server_root)
    print("Scanning folders...", flush=True)
    records = aa.build_records(folders, verbose=args.verbose,
                               with_memory=not args.no_memory)
    print(f"  loaded {len(records):,} records", flush=True)

    succ = gather_successes(records, manifest)
    fails = gather_failures(records, manifest)
    print(f"  {len(succ):,} successful, {len(fails):,} failed")
    if not succ:
        sys.exit("No successful runs to plot.")

    also_png = not args.no_png

    print("\nRendering figures:")
    fig_coverage_failures(records, submitted_idx, manifest, args.out, also_png)
    fig_metric_dists(succ, args.out, also_png)
    fig_flash_threshold(succ, fails, args.out, also_png)
    fig_metric_scatter(succ, args.out, also_png)
    fig_per_board_boxplots(succ, args.out, also_png)

    if not args.no_memory:
        fig_memory_composition(succ, args.out, also_png)
        fig_arena_utilization(succ, args.out, also_png)

    if args.skip_accuracy:
        print("  [skipped fig5_accuracy_pareto via --skip-accuracy]")
    else:
        datasets = [(ds, DATASET_LABEL[ds]) for ds in ("cifar10", "cifar100", "ImageNet16-120")]
        accs = load_accuracies(succ, args.bench, args.cache, datasets)
        if any(accs.values()):
            fig_accuracy_pareto(succ, accs, args.out, also_png)
        else:
            print("  [skipped fig5_accuracy_pareto: no accuracy data]")

    print(f"\nDone. Figures saved to {args.out}/")


if __name__ == "__main__":
    main()
