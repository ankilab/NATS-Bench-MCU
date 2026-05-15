"""Export all data needed to recreate the figures in plot_results.py.

Writes CSV files to export/ (or --out dir):
    hw_metrics.csv    -- one row per successful architecture (figs 2-8)
    failures.csv      -- one row per failed architecture (figs 1, 3)
    coverage.csv      -- pipeline funnel counts (fig 1)
    accuracies.csv    -- test accuracy per dataset joined to hw_metrics (fig 5)
    README.md         -- column descriptions + figure -> columns mapping

Usage:
    python export_plot_data.py [--source server] [--out export/]
    python export_plot_data.py --skip-accuracy   # skip NATS-Bench API lookups
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path



import analyze_artifacts as aa


DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]


def load_accuracies_from_cache(cache_path: Path, indices: list[int]) -> dict[str, dict[int, float]]:
    if not cache_path.exists():
        return {ds: {} for ds in DATASETS}
    cache = json.loads(cache_path.read_text())
    result = {}
    for ds in DATASETS:
        ds_cache = cache.get(ds, {})
        result[ds] = {idx: float(ds_cache[str(idx)]) for idx in indices if str(idx) in ds_cache}
    return result


def load_accuracies_with_api(cache_path: Path, indices: list[int], bench_path: Path) -> dict[str, dict[int, float]]:
    accs = load_accuracies_from_cache(cache_path, indices)
    missing = [idx for idx in indices if any(idx not in accs[ds] for ds in DATASETS)]
    if not missing:
        return accs

    try:
        from nats_bench import create
        api = create(str(bench_path), "tss", fast_mode=True, verbose=False)
    except Exception as e:
        print(f"  [warn] NATS-Bench API unavailable ({e}); using cache only.")
        return accs

    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            pass
    for ds in DATASETS:
        cache.setdefault(ds, {})

    print(f"  Querying NATS-Bench for {len(missing):,} architectures x {len(DATASETS)} datasets...")
    for i, idx in enumerate(sorted(missing)):
        for ds in DATASETS:
            try:
                info = api.get_more_info(idx, ds, hp="200")
                val = float(info.get("test-accuracy"))
                cache[ds][str(idx)] = val
                accs[ds][idx] = val
            except Exception:
                pass
        if (i + 1) % 1000 == 0:
            print(f"    ...{i+1:,}/{len(missing):,}")
            cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))
    print(f"  saved accuracy cache -> {cache_path}")
    return accs


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows):,} rows)")


def write_readme(path: Path, n_succ: int, n_fail: int, n_total: int, has_memory: bool, has_accuracy: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NATS-Bench Hardware Evaluation — Plot Data Export",
        "",
        f"Generated from {n_total:,} nRF5340-DK deployments "
        f"({n_succ:,} successful, {n_fail:,} failed).",
        "",
        "## Files",
        "",
        "### coverage.csv",
        "Pipeline funnel counts. One row per stage.",
        "| column | description |",
        "|--------|-------------|",
        "| stage  | Pipeline stage label |",
        "| count  | Number of architectures reaching this stage |",
        "| pct_of_total | Percentage of the full 15,625-arch search space |",
        "",
        "**Used by:** fig1 (left panel — pipeline funnel bar chart)",
        "",
        "### failures.csv",
        "One row per failed architecture.",
        "| column | description |",
        "|--------|-------------|",
        "| arch_idx | NATS-Bench TSS index (0-based, zero-padded to 5 digits) |",
        "| stage | Failure category: `precheck` (flash overflow) or `exception` |",
        "| int8_kb | INT8 TFLite model file size [KB] from manifest |",
        "| overflow_kb | Model size reported in precheck error message [KB] (NaN if not available) |",
        "| board | Board that ran the request |",
        "",
        "**Used by:** fig1 (right panel — failure breakdown), fig3 (failed arch sizes vs threshold)",
        "",
        "### hw_metrics.csv",
        "One row per successfully deployed architecture. Core hardware measurements.",
        "| column | description |",
        "|--------|-------------|",
        "| arch_idx | NATS-Bench TSS index (0-based) |",
        "| arch_str | Cell topology string, e.g. `|nor~0|+|nor~0|avg~1|+|...` |",
        "| int8_kb | INT8 TFLite model file size [KB] |",
        "| rom_kb | Total on-device ROM (flash) usage [KB] |",
        "| ram_kb | Total on-device RAM usage [KB] (constant 366.3 KB across all archs) |",
        "| latency_s | Mean inference latency [seconds] |",
        "| current_uA | Mean current draw during inference [µA] |",
        "| power_mW | Mean power during inference [mW] |",
        "| energy_mJ | Mean energy per inference [mJ] |",
        "| board | Board unit: nrf5340dk_1 / nrf5340dk_2 / nrf5340dk_3 |",
    ]
    if has_memory:
        lines += [
            "| g_model_kb | Size of the `g_model` symbol in ROM (model weights only) [KB] |",
            "| arena_used_kb | Runtime tensor arena usage reported by TFLM [KB] |",
            "| arena_size_kb | Statically allocated arena size [KB] (constant 350 KB) |",
            "| arena_used_pct | arena_used_kb / arena_size_kb × 100 [%] |",
            "| operators | Number of TFLM operators in the model |",
            '| rom_categories_json | JSON dict of ROM size [KB] per Zephyr memory-report category (keys: `"(no paths)"`, `"WORKSPACE"`, `"ZEPHYR_BASE"`, `"(hidden)"`, etc.) |',
            '| ram_categories_json | JSON dict of RAM size [KB] per Zephyr memory-report category |',
        ]
    lines += [
        "",
        "**Used by:** fig2 (metric distributions), fig3 (successful arch sizes), "
        "fig4 (pairwise scatter), fig6 (per-board boxplots)",
    ]
    if has_memory:
        lines[-1] += ", fig7 (memory composition), fig8 (arena utilization)"
    lines += [
        "",
    ]
    if has_accuracy:
        lines += [
            "### accuracies.csv",
            "Test accuracy from the NATS-Bench simulator for each successfully deployed architecture.",
            "| column | description |",
            "|--------|-------------|",
            "| arch_idx | NATS-Bench TSS index (matches hw_metrics.csv) |",
            "| cifar10 | Test accuracy on CIFAR-10 [%] |",
            "| cifar100 | Test accuracy on CIFAR-100 [%] |",
            "| ImageNet16-120 | Test accuracy on ImageNet-16-120 [%] |",
            "",
            "**Used by:** fig5 (accuracy vs hardware cost Pareto fronts)",
            "",
        ]
    lines += [
        "## Figure → data mapping",
        "",
        "| Figure | File(s) | Key columns |",
        "|--------|---------|-------------|",
        "| fig1 — Pipeline coverage & failures | coverage.csv, failures.csv | count, stage |",
        "| fig2 — HW metric distributions | hw_metrics.csv | rom_kb, latency_s, current_uA, power_mW, energy_mJ |",
        "| fig3 — Flash threshold | hw_metrics.csv, failures.csv | int8_kb, overflow_kb, stage |",
        "| fig4 — Pairwise scatter | hw_metrics.csv | int8_kb, rom_kb, latency_s, energy_mJ |",
        "| fig5 — Accuracy vs cost Pareto | hw_metrics.csv + accuracies.csv | latency_s, energy_mJ, rom_kb, cifar10/100/ImageNet |",
        "| fig6 — Per-board boxplots | hw_metrics.csv | board, power_mW, current_uA, latency_s |",
    ]
    if has_memory:
        lines += [
            "| fig7 — Memory composition | hw_metrics.csv | g_model_kb, rom_*_kb, ram_*_kb |",
            "| fig8 — Arena utilization | hw_metrics.csv | arena_used_kb, arena_size_kb, g_model_kb |",
        ]
    lines += [
        "",
        "## Notes",
        "",
        "- `arch_idx` is the zero-based NATS-Bench TSS index (integer 0–15624).",
        "- RAM is constant at 366.3 KB across all architectures (static tensor arena).",
        "- energy_mJ ≈ power_mW × latency_s (Spearman ρ = 0.999 with measured energy).",
        "- The precheck cutoff in the server pipeline is 800 KB of INT8 model size;",
        "  the empirical flash overflow boundary is ~745 KB (total ROM = INT8 + ~278 KB overhead).",
        "- NATS-Bench paper global best test accuracies: CIFAR-10 94.37%, CIFAR-100 73.51%,",
        "  ImageNet-16-120 47.31% (hp=200 epochs).",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["local", "server"], default="server")
    p.add_argument("--artifacts", default=str(aa.DEFAULT_ARTIFACTS), type=Path)
    p.add_argument("--server-root", default=str(aa.DEFAULT_SERVER_ROOT), type=Path)
    p.add_argument("--manifest", default=str(aa.DEFAULT_MANIFEST), type=Path)
    p.add_argument("--submitted", default=str(aa.DEFAULT_SUBMITTED), type=Path)
    p.add_argument("--bench", default=str(aa.DEFAULT_BENCH), type=Path)
    p.add_argument("--out", default=str(Path(__file__).parent / "export"), type=Path)
    p.add_argument("--cache", default=str(Path(__file__).parent / "figures" / ".accuracy_cache.json"), type=Path)
    p.add_argument("--skip-accuracy", action="store_true")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"Source : {args.source}")
    print(f"Out    : {args.out}")

    manifest = aa.load_manifest(args.manifest)
    submitted_idx = aa.load_submitted(args.submitted)

    folders = aa.discover_local(args.artifacts) if args.source == "local" \
              else aa.discover_server(args.server_root)
    print("Scanning folders...", flush=True)
    records = aa.build_records(folders, verbose=args.verbose, with_memory=not args.no_memory)
    print(f"  loaded {len(records):,} records")

    # ---- coverage.csv ----
    n_manifest = sum(1 for r in manifest.values() if r["status"] == "ok")
    n_sub = len(submitted_idx)
    n_present = len(records)
    n_done = sum(1 for r in records.values() if r["meta_status"] == "done")
    n_error = sum(1 for r in records.values() if r["meta_status"] == "error")
    coverage_rows = [
        {"stage": "NATS-Bench (target)",  "count": aa.NATS_BENCH_TOTAL, "pct_of_total": 100.0},
        {"stage": "TFLite converted",      "count": n_manifest,          "pct_of_total": round(100*n_manifest/aa.NATS_BENCH_TOTAL, 2)},
        {"stage": "Submitted",             "count": n_sub,               "pct_of_total": round(100*n_sub/aa.NATS_BENCH_TOTAL, 2)},
        {"stage": "On disk",               "count": n_present,           "pct_of_total": round(100*n_present/aa.NATS_BENCH_TOTAL, 2)},
        {"stage": "Successful",            "count": n_done,              "pct_of_total": round(100*n_done/aa.NATS_BENCH_TOTAL, 2)},
    ]
    write_csv(args.out / "coverage.csv", ["stage", "count", "pct_of_total"], coverage_rows)

    # ---- failures.csv ----
    fail_rows = []
    for idx, r in records.items():
        if r["meta_status"] != "error":
            continue
        m = manifest.get(idx, {})
        stage = r["fail_stage"]
        if stage == "flash_failed":
            stage = "precheck"
        fail_rows.append({
            "arch_idx": idx,
            "stage": stage,
            "int8_kb": round((m.get("int8_size_bytes") or 0) / 1024.0, 4),
            "overflow_kb": r.get("overflow_kb") if r.get("overflow_kb") is not None else "",
            "board": r.get("board", ""),
        })
    fail_rows.sort(key=lambda r: r["arch_idx"])
    write_csv(args.out / "failures.csv", ["arch_idx", "stage", "int8_kb", "overflow_kb", "board"], fail_rows)

    # ---- hw_metrics.csv ----
    has_memory = not args.no_memory
    hw_rows = []
    succ_indices = []
    for idx, r in records.items():
        res = r.get("results") or {}
        if r["meta_status"] != "done" or not res:
            continue
        m = manifest.get(idx, {})
        uart = r.get("uart") or {}
        ram = r.get("ram") or {}
        rom = r.get("rom") or {}
        ram_cat = (ram.get("by_category") or {}) if ram else {}
        rom_cat = (rom.get("by_category") or {}) if rom else {}
        row = {
            "arch_idx": idx,
            "arch_str": m.get("arch_str", ""),
            "int8_kb": round((m.get("int8_size_bytes") or 0) / 1024.0, 4),
            "rom_kb": round(res.get("rom_usage", 0) / 1024.0, 4),
            "ram_kb": round(res.get("ram_usage", 0) / 1024.0, 4),
            "latency_s": res.get("duration_avg_s", ""),
            "current_uA": res.get("avg_current_uA", ""),
            "power_mW": round((res.get("avg_power_uW") or 0) / 1000.0, 4) if res.get("avg_power_uW") else "",
            "energy_mJ": round((res.get("avg_energy_uJ") or 0) / 1000.0, 4) if res.get("avg_energy_uJ") else "",
            "board": r.get("board", ""),
        }
        if has_memory:
            g_model_b = (rom.get("key_symbols") or {}).get("g_model") or None
            row.update({
                "g_model_kb": round(g_model_b / 1024.0, 4) if g_model_b else "",
                "arena_used_kb": round((uart.get("arena_used") or 0) / 1024.0, 4) if uart.get("arena_used") else "",
                "arena_size_kb": round((uart.get("arena_size") or 0) / 1024.0, 4) if uart.get("arena_size") else "",
                "arena_used_pct": round(uart.get("arena_used_pct"), 2) if uart.get("arena_used_pct") else "",
                "operators": uart.get("operators", ""),
                # Raw category dicts serialised as JSON (bytes); plot_from_export.py
                # applies the same top-5 + relabel logic as plot_results.py.
                "rom_categories_json": json.dumps(
                    {k: round(v / 1024.0, 4) for k, v in rom_cat.items()}) if rom_cat else "",
                "ram_categories_json": json.dumps(
                    {k: round(v / 1024.0, 4) for k, v in ram_cat.items()}) if ram_cat else "",
            })
        hw_rows.append(row)
        succ_indices.append(idx)

    hw_rows.sort(key=lambda r: r["arch_idx"])
    base_cols = ["arch_idx", "arch_str", "int8_kb", "rom_kb", "ram_kb",
                 "latency_s", "current_uA", "power_mW", "energy_mJ", "board"]
    mem_cols = ["g_model_kb", "arena_used_kb", "arena_size_kb", "arena_used_pct", "operators",
                "rom_categories_json", "ram_categories_json"]
    hw_cols = base_cols + (mem_cols if has_memory else [])
    write_csv(args.out / "hw_metrics.csv", hw_cols, hw_rows)

    # ---- accuracies.csv ----
    has_accuracy = False
    if not args.skip_accuracy:
        accs = load_accuracies_with_api(args.cache, succ_indices, args.bench)
        acc_rows = []
        for idx in sorted(succ_indices):
            row = {"arch_idx": idx}
            for ds in DATASETS:
                row[ds] = accs.get(ds, {}).get(idx, "")
            acc_rows.append(row)
        write_csv(args.out / "accuracies.csv", ["arch_idx"] + DATASETS, acc_rows)
        has_accuracy = True

    # ---- README.md ----
    write_readme(
        args.out / "README.md",
        n_succ=n_done, n_fail=n_error,
        n_total=n_done + n_error,
        has_memory=has_memory,
        has_accuracy=has_accuracy,
    )

    print(f"\nDone. All files in {args.out}/")


if __name__ == "__main__":
    main()
