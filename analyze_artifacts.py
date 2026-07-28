"""Analyze the current NATS-Bench hardware-evaluation artifact campaign.

Two data sources are supported:
  --source local   read from hw-nats-bench/artifacts/<arch_idx>/   (default)
  --source server  read directly from the MIMaaS server tree, i.e.
                   <server_root>/user_<N>/request_<request_id>_<timestamp>/

Prints these sections to stdout:
    1.  Pipeline coverage          (manifest -> submitted -> on-disk -> done/error)
    1b. Server <-> CSV consistency check  (server mode only)
    2.  Failure analysis           (failure stages, board cross-tab, flash sizes)
    3.  Hardware-metric distribution (results.json + mean current from ppk2_summary.csv)
    4.  Accuracy <-> hardware join (NATS-Bench API test-acc on three datasets)

stdlib only (+ nats_bench for section 4; section 4 fails soft).
"""

import argparse
import csv
import gzip
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CORRECTIONS = REPO_ROOT / "corrections"
DEFAULT_MANIFEST = REPO_ROOT / "tflite_models" / "manifest.csv"
DEFAULT_SUBMITTED = REPO_ROOT / "submitted_requests.csv"
DEFAULT_BENCH = REPO_ROOT / "benchmarks" / "NATS-tss-v1_0-3ffb9-simple"
DEFAULT_SERVER_ROOT = Path("/home/ankilab/mimaas-server-test/mimaas-server/data/results")

NATS_BENCH_TOTAL = 15625
SERVER_FOLDER_RE = re.compile(r"^request_(\d+)_")
TFLITE_NAME_RE = re.compile(r"nats_bench_model_(\d{5})")

# Mirrors evaluate_nats_models.py so "complete" means the same thing both places.
# The server has used two formats over time:
#   newer : logs stored as .gz, power samples as ppk2_samples.parquet
#   older : uncompressed logs, power samples as ppk2_samples.zip
_SUCCESS_REQUIRED     = {"meta.json", "results.json", "ppk2_summary.csv", "uart.log"}
_SUCCESS_COMPRESSED   = {"flash.log.gz", "model.cpp.gz", "ram.json.gz", "rom.json.gz"}
_SUCCESS_UNCOMPRESSED = {"flash.log", "model.cpp", "ram.json", "rom.json"}
# Legacy name kept for any external callers.
SUCCESS_ARTIFACT_FILES = _SUCCESS_REQUIRED | _SUCCESS_UNCOMPRESSED | {"ppk2_samples.zip"}
FAILURE_ARTIFACT_FILES = {"error.log", "meta.json"}

HW_METRICS = [
    ("ram_usage", "bytes"),
    ("rom_usage", "bytes"),
    ("duration_avg_s", "s"),
    ("avg_current_uA", "uA"),
    ("avg_power_uW", "uW"),
    ("avg_energy_uJ", "uJ"),
]
DATASETS = [("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"), ("ImageNet16-120", "ImageNet-16-120")]


# ---------- small utilities ----------

def section(title):
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def subsection(title):
    print(f"\n--- {title} ---")


def pct(num, denom):
    return f"{100.0 * num / denom:.2f}%" if denom else "n/a"


def fmt_num(x, unit=""):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float):
        if abs(x) >= 1000:
            s = f"{x:,.1f}"
        else:
            s = f"{x:.4f}"
    else:
        s = f"{x:,}"
    return f"{s} {unit}".strip()


def summarize(values):
    """Return (n, mean, std, min, p25, p50, p75, p95, max) for a list of floats."""
    n = len(values)
    if n == 0:
        return (0,) + (None,) * 8
    if n == 1:
        v = values[0]
        return (1, v, 0.0, v, v, v, v, v, v)
    vs = sorted(values)
    mean = statistics.fmean(vs)
    std = statistics.pstdev(vs)
    # Hazen percentiles; quantiles() with n=100 gives 99 cut-points (1..99 percentile).
    qs = statistics.quantiles(vs, n=100, method="inclusive")
    p25, p50, p75, p95 = qs[24], qs[49], qs[74], qs[94]
    return (n, mean, std, vs[0], p25, p50, p75, p95, vs[-1])


def print_stats_table(rows):
    header = ["metric", "N", "mean", "std", "min", "p25", "p50", "p75", "p95", "max"]
    widths = [22, 5, 14, 14, 14, 14, 14, 14, 14, 14]
    line = "  ".join(f"{h:>{w}}" for h, w in zip(header, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        cells = [f"{r[i]:>{widths[i]}}" if i in (0, 1) else f"{r[i]:>{widths[i]}}" for i in range(len(r))]
        print("  ".join(cells))


def ranks(values):
    """Return ranks (average ties) for use in Spearman correlation."""
    pairs = sorted(enumerate(values), key=lambda p: p[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[pairs[k][0]] = avg_rank
        i = j + 1
    return out


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    try:
        return statistics.correlation(rx, ry)
    except statistics.StatisticsError:
        return None


# ---------- data loading ----------

def load_manifest(path):
    """Return {index: {arch_str, int8_size_bytes, status}}."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row["index"])
            except (KeyError, ValueError):
                continue
            try:
                int8_size = int(row.get("int8_size_bytes") or 0)
            except ValueError:
                int8_size = 0
            out[idx] = {
                "arch_str": row.get("arch_str", ""),
                "int8_size_bytes": int8_size,
                "status": row.get("status", ""),
            }
    return out


def load_submitted(path):
    """Return set of architecture indices that have been submitted to MIMaaS."""
    out = set()
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            model = row.get("model", "")
            m = re.search(r"nats_bench_model_(\d{5})", model)
            if m:
                out.add(int(m.group(1)))
    return out


def discover_local(artifacts_dir):
    """Yield (arch_index, Path) for every artifacts/XXXXX/ subdir with meta.json.

    Local layout: artifacts/<5-digit-arch-index>/{meta.json, results.json, ...}
    """
    if not artifacts_dir.is_dir():
        return
    for entry in sorted(artifacts_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            idx = int(entry.name)
        except ValueError:
            continue
        if (entry / "meta.json").is_file():
            yield idx, entry


def discover_corrections(corrections_dir):
    """Yield (arch_index, Path) for every corrections/XXXXX/ subdir with meta.json.

    Same layout as artifacts/ (one folder per zero-padded arch index). Applied
    as an authoritative overlay on top of the released artifacts; see the
    `corrections_iter` argument of build_records() and corrections/README.md.
    """
    if not corrections_dir.is_dir():
        return
    for entry in sorted(corrections_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            idx = int(entry.name)
        except ValueError:
            continue
        if (entry / "meta.json").is_file():
            yield idx, entry


def discover_server(server_root):
    """Yield (arch_index, Path) for every user_*/request_<id>_<ts>/ folder.

    Server layout: <server_root>/user_<N>/request_<request_id>_<timestamp>/{...}
    The architecture index is parsed from meta.json's `network_file_name` (or
    falls back to the tflite filename in the folder). Yields nothing if a folder
    has no parseable index — those are reported separately by the caller.
    """
    if not server_root.is_dir():
        return
    for user_dir in sorted(server_root.iterdir()):
        if not user_dir.is_dir() or not user_dir.name.startswith("user_"):
            continue
        for entry in sorted(user_dir.iterdir()):
            if not entry.is_dir() or not SERVER_FOLDER_RE.match(entry.name):
                continue
            if not (entry / "meta.json").is_file():
                continue
            idx = _extract_arch_index(entry)
            if idx is not None:
                yield idx, entry


def _extract_arch_index(folder):
    """Return arch index from meta.json/network_file_name or any tflite in folder."""
    meta = read_json(folder / "meta.json") or {}
    name = meta.get("network_file_name") or ""
    m = TFLITE_NAME_RE.search(name)
    if m:
        return int(m.group(1))
    try:
        for fn in os.listdir(folder):
            m = TFLITE_NAME_RE.search(fn)
            if m and fn.endswith(".tflite"):
                return int(m.group(1))
    except OSError:
        pass
    return None


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        # Fall back to a gzipped sibling (the server compresses the larger
        # reports, e.g. ram.json.gz / rom.json.gz, to save space).
        try:
            with gzip.open(f"{path}.gz", "rt") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    except (OSError, json.JSONDecodeError):
        return None


def mean_current_from_ppk2(csv_path):
    """Return mean(avg_current_uA) across segments, or None if unavailable."""
    if not csv_path.is_file():
        return None
    vals = []
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    vals.append(float(row["avg_current_uA"]))
                except (KeyError, ValueError):
                    continue
    except OSError:
        return None
    return statistics.fmean(vals) if vals else None


_FAILED_STAGE_RE = re.compile(r"^Failed Stage:\s*(.+)$", re.M)
_ERROR_BODY_RE = re.compile(r"^Error:\s*\n(.+)", re.M)
_FLASH_OVERFLOW_RE = re.compile(r"Model file \((\d+)\s*KB\) exceeds")

_UART_OPS_RE = re.compile(r"\bOperators:\s*(\d+)")
_UART_ARENA_SIZE_RE = re.compile(r"Arena size\s*:\s*(\d+)\s*bytes")
_UART_ARENA_USED_RE = re.compile(r"Used bytes\s*:\s*(\d+)\s*bytes\s*\((\d+)%\)")


def parse_uart_log(path):
    """Extract operators, arena_size, arena_used from a TFLM uart.log.

    The arena is statically allocated in main.c (currently 350 * 1024 bytes)
    and the interpreter prints `arena_used_bytes()` after AllocateTensors().
    """
    if not path.is_file():
        return None
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    out = {}
    m = _UART_OPS_RE.search(text)
    if m: out["operators"] = int(m.group(1))
    m = _UART_ARENA_SIZE_RE.search(text)
    if m: out["arena_size"] = int(m.group(1))
    m = _UART_ARENA_USED_RE.search(text)
    if m:
        out["arena_used"] = int(m.group(1))
        out["arena_used_pct"] = int(m.group(2))
    return out or None


def summarize_memory_tree(path, key_symbols=None):
    """Walk a Zephyr west ram.json / rom.json tree and return a small summary.

    Returns {total, by_category, key_symbols} where:
      - by_category: top-level child sizes (e.g. '(no paths)', 'ZEPHYR_BASE',
        'WORKSPACE', '(hidden)').
      - key_symbols: sizes of named symbols anywhere in the tree, matched by
        EXACT symbol name (the `name` field, not the identifier path).

    `key_symbols` is a dict mapping output label -> exact symbol name to find.
    Example: {"tensor_arena": "_ZN12_GLOBAL__N_1L12tensor_arenaE",
              "g_model":     "g_model"}
    """
    data = read_json(path)
    if not data or "symbols" not in data:
        return None
    root = data["symbols"]
    out = {"total": data.get("total_size", root.get("size", 0)),
           "by_category": {}, "key_symbols": {}}
    for child in root.get("children", []) or []:
        out["by_category"][child.get("name", "?")] = int(child.get("size", 0))
    if key_symbols:
        # invert: name -> label, so we can lookup in O(1) during the walk
        name_to_label = {sym_name: label for label, sym_name in key_symbols.items()}
        stack = [root]
        while stack and len(out["key_symbols"]) < len(key_symbols):
            n = stack.pop()
            label = name_to_label.get(n.get("name") or "")
            if label and label not in out["key_symbols"]:
                out["key_symbols"][label] = int(n.get("size", 0))
            stack.extend(n.get("children", []) or [])
    return out


def parse_error_log(path):
    """Return (stage, first_error_line, flash_overflow_kb_or_None)."""
    try:
        text = path.read_text()
    except OSError:
        return ("unknown", "", None)
    stage = "unknown"
    m = _FAILED_STAGE_RE.search(text)
    if m:
        stage = m.group(1).strip()
    body = ""
    m = _ERROR_BODY_RE.search(text)
    if m:
        body = m.group(1).strip().splitlines()[0] if m.group(1).strip() else ""
    overflow_kb = None
    m = _FLASH_OVERFLOW_RE.search(text)
    if m:
        overflow_kb = int(m.group(1))
    return (stage, body, overflow_kb)


# ---------- sections ----------

def section_coverage(manifest, submitted_idx, artifact_records, source, source_label):
    section("1. Pipeline coverage")
    n_manifest_ok = sum(1 for r in manifest.values() if r["status"] == "ok")
    n_submitted = len(submitted_idx)
    n_present = len(artifact_records)
    n_done = sum(1 for r in artifact_records.values() if r["meta_status"] == "done")
    n_error = sum(1 for r in artifact_records.values() if r["meta_status"] == "error")
    n_other = n_present - n_done - n_error
    n_terminal_success = sum(
        1 for r in artifact_records.values()
        if r["meta_status"] == "done" and r["complete_success"]
    )
    n_terminal_failure = sum(
        1 for r in artifact_records.values()
        if r["meta_status"] == "error" and r["complete_failure"]
    )
    presence_label = f"Folders found ({source_label})"

    rows = [
        ("Architectures in NATS-Bench (target)", NATS_BENCH_TOTAL, NATS_BENCH_TOTAL),
        ("TFLite-converted (manifest status=ok)", n_manifest_ok, NATS_BENCH_TOTAL),
        ("Submitted to MIMaaS", n_submitted, NATS_BENCH_TOTAL),
        (presence_label, n_present, NATS_BENCH_TOTAL),
        ("  - meta.status == done", n_done, n_present),
        ("    of which file-complete (all 9 success files)", n_terminal_success, n_done),
        ("  - meta.status == error", n_error, n_present),
        ("    of which file-complete (error.log + meta.json)", n_terminal_failure, n_error),
        ("  - other / partial", n_other, n_present),
    ]
    label_w = max(len(r[0]) for r in rows)
    for label, count, denom in rows:
        print(f"  {label:<{label_w}}  {count:>7,}  ({pct(count, denom)} of {denom:,})")

    not_submitted = NATS_BENCH_TOTAL - n_submitted
    presence_gap = n_submitted - n_present
    presence_gap_label = (
        "Submitted but not yet on server"
        if source == "server" else "Submitted but not yet downloaded"
    )
    print()
    print(f"  Remaining to submit                  : {not_submitted:>7,}")
    print(f"  {presence_gap_label:<37}: {presence_gap:>7,}")


def section_failures(artifact_records, manifest):
    section("2. Failure analysis")
    failed = {idx: r for idx, r in artifact_records.items() if r["meta_status"] == "error"}
    print(f"  Total failed evaluations: {len(failed)}")
    if not failed:
        return

    stage_counts = Counter(r["fail_stage"] for r in failed.values())
    subsection("Failures by stage")
    for stage, n in stage_counts.most_common():
        print(f"  {stage:<24} {n:>5}  ({pct(n, len(failed))})")

    subsection("Example error messages per stage")
    seen_stages = set()
    for idx, r in failed.items():
        s = r["fail_stage"]
        if s in seen_stages or not r["fail_msg"]:
            continue
        seen_stages.add(s)
        print(f"  [{s}] idx={idx:05d}: {r['fail_msg'][:140]}")

    subsection("All exception failures (full detail)")
    exception_records = {idx: r for idx, r in failed.items() if r["fail_stage"] == "exception"}
    if exception_records:
        for idx, r in sorted(exception_records.items()):
            print(f"  idx={idx:05d}  request_id={r['request_id']}  board={r['board']}")
            print(f"    msg: {r['fail_msg'][:200]}")
    else:
        print("  (none)")

    board_x_stage = defaultdict(Counter)
    for r in failed.values():
        board_x_stage[r["board"] or "unknown"][r["fail_stage"]] += 1
    subsection("Failures cross-tabulated by board x stage")
    all_stages = sorted({s for c in board_x_stage.values() for s in c})
    boards = sorted(board_x_stage)
    header_w = max((len(b) for b in boards), default=8)
    print("  " + "stage".ljust(24) + "  " + "  ".join(b.ljust(header_w) for b in boards) + "  total")
    for s in all_stages:
        row_total = sum(board_x_stage[b][s] for b in boards)
        cells = "  ".join(str(board_x_stage[b][s]).ljust(header_w) for b in boards)
        print(f"  {s.ljust(24)}  {cells}  {row_total}")

    overflow_sizes = [r["overflow_kb"] for r in failed.values() if r["overflow_kb"] is not None]
    if overflow_sizes:
        subsection("Flash-overflow (precheck) reported model sizes [KB]")
        stats = summarize([float(v) for v in overflow_sizes])
        print(f"  N={stats[0]}  mean={fmt_num(stats[1])}  std={fmt_num(stats[2])}  "
              f"min={fmt_num(stats[3])}  p50={fmt_num(stats[5])}  max={fmt_num(stats[8])}")

    subsection("INT8 size (from manifest) for failed architectures [KB]")
    failed_int8_kb = [
        manifest.get(idx, {}).get("int8_size_bytes", 0) / 1024.0
        for idx in failed
        if manifest.get(idx, {}).get("int8_size_bytes")
    ]
    if failed_int8_kb:
        stats = summarize(failed_int8_kb)
        print(f"  N={stats[0]}  mean={fmt_num(stats[1])}  std={fmt_num(stats[2])}  "
              f"min={fmt_num(stats[3])}  p50={fmt_num(stats[5])}  max={fmt_num(stats[8])}")
        success_int8_kb = [
            manifest.get(idx, {}).get("int8_size_bytes", 0) / 1024.0
            for idx, r in artifact_records.items()
            if r["meta_status"] == "done" and manifest.get(idx, {}).get("int8_size_bytes")
        ]
        if success_int8_kb:
            ss = summarize(success_int8_kb)
            print(f"  (for comparison, successful INT8 sizes: "
                  f"N={ss[0]}  mean={fmt_num(ss[1])}  p50={fmt_num(ss[5])}  max={fmt_num(ss[8])})")


def section_metrics(artifact_records):
    section("3. Hardware-metric distribution (successful runs)")
    successes = {idx: r for idx, r in artifact_records.items() if r["meta_status"] == "done" and r["results"]}
    print(f"  Architectures with parseable results.json: {len(successes)}")
    if not successes:
        return

    metric_values = {name: [] for name, _ in HW_METRICS}
    for r in successes.values():
        for name, _ in HW_METRICS:
            v = r["results"].get(name)
            if v is not None:
                metric_values[name].append(float(v))

    subsection("Per-metric distribution")
    rows = []
    for name, unit in HW_METRICS:
        n, mean, std, mn, p25, p50, p75, p95, mx = summarize(metric_values[name])
        rows.append((
            f"{name} [{unit}]", str(n),
            fmt_num(mean), fmt_num(std), fmt_num(mn),
            fmt_num(p25), fmt_num(p50), fmt_num(p75), fmt_num(p95), fmt_num(mx),
        ))
    print_stats_table(rows)

    subsection("Near-constant fields (coefficient of variation < 1%)")
    flagged = []
    for name, _ in HW_METRICS:
        vs = metric_values[name]
        if len(vs) < 2:
            continue
        mean = statistics.fmean(vs)
        std = statistics.pstdev(vs)
        cv = std / mean if mean else 0
        if cv < 0.01:
            flagged.append((name, mean, std, cv, min(vs), max(vs)))
    if not flagged:
        print("  (none)")
    for name, mean, std, cv, mn, mx in flagged:
        print(f"  {name:<20} mean={fmt_num(mean)}  std={fmt_num(std)}  "
              f"CV={cv*100:.3f}%  range=[{fmt_num(mn)}, {fmt_num(mx)}]")
    if flagged:
        print("  -> these fields are effectively a single platform constant in this dataset;")
        print("     the paper should describe them as fixed platform properties, not searchable.")

    subsection("Board distribution among successful runs")
    by_board = defaultdict(list)
    for r in successes.values():
        by_board[r["board"] or "unknown"].append(r["results"])
    for board, rs in sorted(by_board.items()):
        print(f"  {board}: {len(rs)} runs")

    subsection("Per-board mean (power / current / latency) for systematic-bias check")
    metric_keys = ["avg_power_uW", "avg_current_uA", "duration_avg_s"]
    label_w = 18
    print("  " + "board".ljust(label_w) + "  N    " + "  ".join(m.ljust(20) for m in metric_keys))
    for board, rs in sorted(by_board.items()):
        cells = []
        for k in metric_keys:
            xs = [float(x[k]) for x in rs if x.get(k) is not None]
            if not xs:
                cells.append("n/a".ljust(20))
            else:
                mean = statistics.fmean(xs)
                std = statistics.pstdev(xs) if len(xs) > 1 else 0
                cells.append(f"{mean:,.1f}±{std:,.1f}".ljust(20))
        print(f"  {board.ljust(label_w)}  {len(rs):<4} " + "  ".join(cells))


def section_memory(records):
    section("5. Memory layout (RAM / ROM / tensor arena)")
    succ = [r for r in records.values() if r["meta_status"] == "done"]
    have_ram = [r for r in succ if r.get("ram")]
    have_rom = [r for r in succ if r.get("rom")]
    have_uart = [r for r in succ if r.get("uart") and r["uart"].get("arena_used")]
    if not have_ram and not have_rom and not have_uart:
        print("  [skip] no ram/rom/uart data parsed (re-run with --with-memory).")
        return
    print(f"  Records with ram.json     : {len(have_ram):,}")
    print(f"  Records with rom.json     : {len(have_rom):,}")
    print(f"  Records with uart arena info: {len(have_uart):,}")

    # Tensor arena: allocated vs. actually used
    if have_uart:
        sizes = [r["uart"]["arena_size"] for r in have_uart if "arena_size" in r["uart"]]
        useds = [r["uart"]["arena_used"] for r in have_uart]
        pcts = [100.0 * u / s for u, s in zip(useds, sizes) if s]
        ops = [r["uart"]["operators"] for r in have_uart if "operators" in r["uart"]]
        subsection("Tensor arena utilization")
        arena_sizes = set(sizes)
        if len(arena_sizes) == 1:
            print(f"  Arena (statically allocated): {next(iter(arena_sizes)):,} bytes "
                  f"= {next(iter(arena_sizes))/1024:.1f} KB  (constant across runs)")
        else:
            print(f"  Arena size varies: min={min(sizes):,} max={max(sizes):,} bytes")
        u_stats = summarize([float(u) for u in useds])
        p_stats = summarize(pcts)
        print(f"  Used [bytes]: N={u_stats[0]:,}  mean={fmt_num(u_stats[1])}  "
              f"std={fmt_num(u_stats[2])}  min={fmt_num(u_stats[3])}  "
              f"p50={fmt_num(u_stats[5])}  p95={fmt_num(u_stats[7])}  max={fmt_num(u_stats[8])}")
        print(f"  Used [%]    : mean={p_stats[1]:.1f}%  median={p_stats[5]:.1f}%  "
              f"min={p_stats[3]:.1f}%  max={p_stats[8]:.1f}%")
        headroom_max = max(sizes) - max(useds)
        print(f"  Worst-case headroom: arena_size - max(used) = "
              f"{max(sizes)-max(useds):,} bytes ({headroom_max/max(sizes)*100:.0f}% of arena)")
        if ops:
            o_stats = summarize([float(o) for o in ops])
            print(f"  Operators per model: N={o_stats[0]:,} mean={o_stats[1]:.1f} "
                  f"median={o_stats[5]:.0f} min={o_stats[3]:.0f} max={o_stats[8]:.0f}")

    # ROM composition: where do those ~650 KB go?
    if have_rom:
        subsection("ROM composition (means across successful runs)")
        cat_totals = {}
        for r in have_rom:
            for cat, sz in r["rom"]["by_category"].items():
                cat_totals.setdefault(cat, []).append(sz)
        rom_totals = [r["rom"]["total"] for r in have_rom]
        mean_total = statistics.fmean(rom_totals)
        label_w = max((len(c) for c in cat_totals), default=10)
        print(f"  {'category'.ljust(label_w)}  {'mean bytes':>12}  {'mean %':>7}  {'min %':>7}  {'max %':>7}")
        for cat, vals in sorted(cat_totals.items(), key=lambda kv: -statistics.fmean(kv[1])):
            mean_b = statistics.fmean(vals)
            pcts = [100.0 * v / r["rom"]["total"] for v, r in zip(vals, have_rom) if r["rom"]["total"]]
            print(f"  {cat.ljust(label_w)}  {mean_b:>12,.0f}  "
                  f"{statistics.fmean(pcts):>6.1f}%  "
                  f"{min(pcts):>6.1f}%  {max(pcts):>6.1f}%")
        # g_model share
        gm = [r["rom"]["key_symbols"].get("tensor_arena", 0) for r in have_rom]  # noqa - not used
        g_model_sizes = [r["rom"]["key_symbols"].get("g_model") for r in have_rom
                         if r["rom"]["key_symbols"].get("g_model")]
        if g_model_sizes:
            gm_stats = summarize([float(x) for x in g_model_sizes])
            mean_share = statistics.fmean([100.0 * g / r["rom"]["total"]
                                           for g, r in zip(g_model_sizes, have_rom)
                                           if r["rom"]["total"]])
            print(f"  g_model symbol: N={gm_stats[0]:,}  "
                  f"mean={fmt_num(gm_stats[1])} bytes  "
                  f"p50={fmt_num(gm_stats[5])}  max={fmt_num(gm_stats[8])}  "
                  f"mean_share_of_ROM={mean_share:.1f}%")

    # RAM composition
    if have_ram:
        subsection("RAM composition (means across successful runs)")
        cat_totals = {}
        for r in have_ram:
            for cat, sz in r["ram"]["by_category"].items():
                cat_totals.setdefault(cat, []).append(sz)
        label_w = max((len(c) for c in cat_totals), default=10)
        print(f"  {'category'.ljust(label_w)}  {'mean bytes':>12}  {'mean %':>7}")
        for cat, vals in sorted(cat_totals.items(), key=lambda kv: -statistics.fmean(kv[1])):
            mean_b = statistics.fmean(vals)
            pcts = [100.0 * v / r["ram"]["total"] for v, r in zip(vals, have_ram) if r["ram"]["total"]]
            print(f"  {cat.ljust(label_w)}  {mean_b:>12,.0f}  {statistics.fmean(pcts):>6.1f}%")


def section_accuracy(artifact_records, manifest, bench_path):
    section("4. Accuracy <-> hardware join (NATS-Bench API)")
    try:
        from nats_bench import create
    except ImportError as e:
        print(f"  [skip] could not import nats_bench: {e}")
        return
    if not Path(bench_path).exists():
        print(f"  [skip] benchmark path not found: {bench_path}")
        return
    try:
        api = create(str(bench_path), "tss", fast_mode=True, verbose=False)
    except Exception as e:  # noqa: BLE001 -- API can raise many things
        print(f"  [skip] could not load NATS-Bench API: {e}")
        return

    successes = {idx: r for idx, r in artifact_records.items() if r["meta_status"] == "done" and r["results"]}
    if not successes:
        print("  No successful runs to join.")
        return

    rows = []
    print(f"  Querying NATS-Bench (hp='200') for {len(successes)} architectures...", flush=True)
    failed_lookups = 0
    for idx in sorted(successes):
        row = {"index": idx, **successes[idx]["results"]}
        ok = True
        for key, _ in DATASETS:
            try:
                info = api.get_more_info(idx, key, hp="200")
                row[f"acc_{key}"] = float(info.get("test-accuracy"))
            except Exception:  # noqa: BLE001
                row[f"acc_{key}"] = None
                ok = False
        if not ok:
            failed_lookups += 1
        rows.append(row)
    if failed_lookups:
        print(f"  warning: {failed_lookups} architectures had at least one missing accuracy")

    subsection("Spearman rank correlation: accuracy vs. hardware cost")
    metric_keys = ["rom_usage", "duration_avg_s", "avg_current_uA", "avg_power_uW", "avg_energy_uJ"]
    header = "  " + "dataset".ljust(20) + "  " + "  ".join(m.ljust(18) for m in metric_keys)
    print(header)
    for key, label in DATASETS:
        cells = []
        for m in metric_keys:
            xs, ys = [], []
            for r in rows:
                if r.get(f"acc_{key}") is None or r.get(m) is None:
                    continue
                xs.append(r[f"acc_{key}"])
                ys.append(r[m])
            rho = spearman(xs, ys)
            cells.append((f"{rho:+.3f} (N={len(xs)})" if rho is not None else "n/a").ljust(18))
        print(f"  {label.ljust(20)}  " + "  ".join(cells))

    subsection("Best deployed accuracy vs. NATS-Bench paper-reported optimum")
    paper_optima = {"cifar10": 94.37, "cifar100": 73.51, "ImageNet16-120": 47.31}
    for key, label in DATASETS:
        accs = [r[f"acc_{key}"] for r in rows if r.get(f"acc_{key}") is not None]
        if not accs:
            print(f"  {label}: no data")
            continue
        best = max(accs)
        opt = paper_optima.get(key)
        gap = (opt - best) if opt is not None else None
        direction = "below" if gap is not None and gap >= 0 else "above"
        gap_str = f"{abs(gap):.2f} pp {direction}" if gap is not None else "n/a"
        print(f"  {label}: best deployed = {best:.2f}%  "
              f"(paper optimum = {opt}%, deployed-best is {gap_str}; N={len(accs)})")

    subsection("Top-5 Pareto-optimal architectures (CIFAR-10 test-acc vs. cost)")

    def pareto_front(key, cost_key, lower_cost_better=True):
        pts = [(r["index"], r.get(f"acc_{key}"), r.get(cost_key)) for r in rows]
        pts = [p for p in pts if p[1] is not None and p[2] is not None]
        front = []
        for idx, acc, cost in pts:
            dominated = False
            for _, a2, c2 in pts:
                better_cost = (c2 <= cost) if lower_cost_better else (c2 >= cost)
                strictly = (a2 > acc and (c2 < cost if lower_cost_better else c2 > cost)) or \
                           (a2 >= acc and (c2 < cost if lower_cost_better else c2 > cost))
                if a2 >= acc and better_cost and strictly:
                    dominated = True
                    break
            if not dominated:
                front.append((idx, acc, cost))
        front.sort(key=lambda t: t[1], reverse=True)
        return front[:5]

    for cost_key, cost_label in [
        ("duration_avg_s", "latency [s]"),
        ("avg_energy_uJ", "energy [uJ]"),
        ("rom_usage", "rom [bytes]"),
    ]:
        print(f"\n  Pareto front -- CIFAR-10 test-acc vs. {cost_label}:")
        front = pareto_front("cifar10", cost_key, lower_cost_better=True)
        if not front:
            print("    (empty)")
            continue
        for idx, acc, cost in front:
            arch = manifest.get(idx, {}).get("arch_str", "")
            print(f"    idx={idx:05d}  acc={acc:6.2f}%  {cost_label}={fmt_num(cost)}  arch={arch[:80]}")


# ---------- main ----------

def _make_record(idx, d, with_memory=False):
    """Build a single {field: value} record from one artifact folder."""
    meta = read_json(d / "meta.json") or {}
    results = read_json(d / "results.json")
    current = mean_current_from_ppk2(d / "ppk2_summary.csv")
    if results is not None and current is not None:
        results = dict(results)
        results["avg_current_uA"] = current
    uart = None
    ram = None
    rom = None
    if with_memory and meta.get("status") == "done":
        uart = parse_uart_log(d / "uart.log")
        ram = summarize_memory_tree(
            d / "ram.json",
            key_symbols={"tensor_arena": "_ZN12_GLOBAL__N_1L12tensor_arenaE"},
        )
        rom = summarize_memory_tree(
            d / "rom.json",
            key_symbols={"g_model": "g_model"},
        )
    try:
        present = set(os.listdir(d))
    except OSError:
        present = set()
    tflite_name = meta.get("network_file_name") or f"nats_bench_model_{idx:05d}_int8.tflite"
    has_required = _SUCCESS_REQUIRED | {tflite_name} <= present
    has_power    = "ppk2_samples.zip" in present or "ppk2_samples.parquet" in present
    logs_ok      = _SUCCESS_COMPRESSED <= present or _SUCCESS_UNCOMPRESSED <= present
    complete_success = has_required and has_power and logs_ok
    complete_failure = FAILURE_ARTIFACT_FILES | {tflite_name} <= present
    fail_stage, fail_msg, overflow_kb = ("", "", None)
    if meta.get("status") == "error":
        fail_stage, fail_msg, overflow_kb = parse_error_log(d / "error.log")
    return {
        "meta": meta,
        "meta_status": meta.get("status", ""),
        "board": meta.get("board", ""),
        "request_id": meta.get("request_id"),
        "submitted_at": meta.get("submitted_at", ""),
        "folder": str(d),
        "results": results,
        "complete_success": complete_success,
        "complete_failure": complete_failure,
        # All flash-infeasibility rejections share the single `precheck` label
        # (the corrected precheck gates on the exact 746 KB flash budget, so
        # the campaign's later link-stage `flash_failed` cases fold in here).
        "fail_stage": "precheck" if (fail_stage or "unknown") == "flash_failed" else (fail_stage or "unknown"),
        "fail_msg": fail_msg,
        "overflow_kb": overflow_kb,
        "uart": uart,
        "ram": ram,
        "rom": rom,
    }


def build_records(folder_iter, verbose=False, with_memory=False, corrections_iter=None):
    """Build {arch_index: record} from any (idx, Path) iterator.

    Handles duplicate indices (e.g. an architecture submitted twice -> two
    server folders): keeps the one with status==done if either has it, else
    the most recently submitted (alphabetic on submitted_at suffix is fine
    because the timestamp is ISO-like in the folder name).

    If with_memory=True, also parses uart.log, ram.json, rom.json for each
    successful record. Adds ~1 min to the scan when running over the full
    server tree but unlocks the memory-layout analysis section.

    corrections_iter, if given, is a second (idx, Path) iterator applied last as
    an authoritative overlay: each record unconditionally overrides the base
    record for that index, EXCEPT a folder whose meta.json status is `pending`
    or `excluded`, which removes the index entirely (used for architectures
    awaiting a hardware remeasurement). See corrections/README.md for provenance.
    """
    records = {}
    duplicates = 0
    processed = 0
    for idx, d in folder_iter:
        new_rec = _make_record(idx, d, with_memory)
        if idx in records:
            duplicates += 1
            old = records[idx]
            def _rank(s): return 2 if s == "done" else 1 if s == "error" else 0
            keep_new = (
                _rank(new_rec["meta_status"]) > _rank(old["meta_status"])
                or (_rank(new_rec["meta_status"]) == _rank(old["meta_status"])
                    and new_rec["submitted_at"] > old["submitted_at"])
            )
            if keep_new:
                records[idx] = new_rec
        else:
            records[idx] = new_rec
        processed += 1
        if verbose and processed % 2000 == 0:
            print(f"  ...processed {processed:,} folders", flush=True)
    if duplicates:
        print(f"  note: {duplicates} duplicate submissions of the same architecture were "
              f"collapsed (kept the done/most-recent record per index)")

    if corrections_iter is not None:
        n_override, n_excluded = 0, 0
        for idx, d in corrections_iter:
            meta = read_json(d / "meta.json") or {}
            if meta.get("status") in ("pending", "excluded"):
                if records.pop(idx, None) is not None:
                    n_excluded += 1
                continue
            records[idx] = _make_record(idx, d, with_memory)
            n_override += 1
        if n_override or n_excluded:
            print(f"  corrections overlay: {n_override} record(s) overridden, "
                  f"{n_excluded} index(es) excluded (pending remeasurement)")
    return records


def server_consistency_check(server_root, csv_path, records):
    """Cross-verify server folders <-> submitted_requests.csv.

    Reports counts and confirms that the architecture index inside each server
    folder agrees with the index implied by the CSV's model column.
    """
    section("1b. Server <-> CSV consistency check")
    if not csv_path.exists():
        print(f"  [skip] submitted_requests.csv not found: {csv_path}")
        return

    csv_rid_to_idx = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rid = int(row["request_id"])
            except (KeyError, ValueError):
                continue
            m = TFLITE_NAME_RE.search(row.get("model", ""))
            if m:
                csv_rid_to_idx[rid] = int(m.group(1))

    server_rid_to_idx = {}
    user_counts = Counter()
    if server_root.is_dir():
        for user_dir in sorted(server_root.iterdir()):
            if not user_dir.is_dir() or not user_dir.name.startswith("user_"):
                continue
            for entry in user_dir.iterdir():
                m = SERVER_FOLDER_RE.match(entry.name)
                if not m or not entry.is_dir():
                    continue
                rid = int(m.group(1))
                idx = _extract_arch_index(entry)
                if idx is not None:
                    server_rid_to_idx[rid] = idx
                    user_counts[user_dir.name] += 1

    inter = set(csv_rid_to_idx) & set(server_rid_to_idx)
    csv_only = set(csv_rid_to_idx) - set(server_rid_to_idx)
    srv_only = set(server_rid_to_idx) - set(csv_rid_to_idx)
    mismatches = [rid for rid in inter if csv_rid_to_idx[rid] != server_rid_to_idx[rid]]

    print(f"  Server folders (across users) : {len(server_rid_to_idx):>7,}")
    print(f"  CSV submission rows           : {len(csv_rid_to_idx):>7,}")
    print(f"  Intersection (in both)        : {len(inter):>7,}")
    print(f"  CSV-only (submitted, no folder): {len(csv_only):>7,}")
    print(f"  Server-only (folder, no CSV)  : {len(srv_only):>7,}")
    print(f"  Folder<->CSV index mismatches : {len(mismatches):>7,}"
          f"   {'<-- INVESTIGATE' if mismatches else '(perfect)'}")

    if user_counts:
        subsection("Per-user folder counts")
        for u, n in user_counts.most_common():
            print(f"  {u:<12} {n:,} folders")

    if csv_only:
        srt = sorted(csv_only)
        print(f"\n  CSV-only id range: {min(srt)} .. {max(srt)}  (likely queued or not yet executed)")
    if srv_only:
        srt = sorted(srv_only)
        print(f"  Server-only id range: {min(srt)} .. {max(srt)}  "
              f"(likely pre-CSV submissions or from another campaign)")
    if mismatches[:5]:
        print("\n  Sample mismatches (request_id: csv_idx vs server_idx):")
        for rid in mismatches[:5]:
            print(f"    {rid}: {csv_rid_to_idx[rid]:05d} vs {server_rid_to_idx[rid]:05d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["local", "server"], default="local",
                        help="Read artifacts from local hw-nats-bench/artifacts/ "
                             "(default) or directly from the MIMaaS server folder.")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS), type=Path,
                        help="Local artifacts directory (for --source local).")
    parser.add_argument("--server-root", default=str(DEFAULT_SERVER_ROOT), type=Path,
                        help="MIMaaS server data/results root containing user_<N>/ "
                             "subdirs (for --source server).")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=Path)
    parser.add_argument("--submitted", default=str(DEFAULT_SUBMITTED), type=Path)
    parser.add_argument("--bench", default=str(DEFAULT_BENCH), type=Path,
                        help="Path to NATS-tss-v1_0-3ffb9-simple (per-arch pickle dir)")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip section 4 (NATS-Bench accuracy join)")
    parser.add_argument("--with-memory", action="store_true",
                        help="Parse ram.json / rom.json / uart.log and emit section 5 "
                             "(memory layout & arena utilization). Adds ~1 min to scanning.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress while scanning many folders.")
    args = parser.parse_args()

    print(f"Repo root           : {REPO_ROOT}")
    print(f"Source              : {args.source}")
    if args.source == "local":
        print(f"Artifacts directory : {args.artifacts}")
    else:
        print(f"Server root         : {args.server_root}")
    print(f"Manifest CSV        : {args.manifest}")
    print(f"Submitted CSV       : {args.submitted}")
    print(f"NATS-Bench path     : {args.bench}")

    if not args.manifest.exists():
        sys.exit(f"manifest not found: {args.manifest}")

    manifest = load_manifest(args.manifest)
    submitted_idx = load_submitted(args.submitted)

    if args.source == "local":
        folders = discover_local(args.artifacts)
        source_label = "local artifacts/"
    else:
        if not args.server_root.is_dir():
            sys.exit(f"server root not found: {args.server_root}")
        folders = discover_server(args.server_root)
        source_label = "server user_*/request_*"

    print("\nScanning folders...", flush=True)
    records = build_records(folders, verbose=args.verbose, with_memory=args.with_memory)
    print(f"  loaded {len(records):,} unique-architecture records", flush=True)

    section_coverage(manifest, submitted_idx, records, args.source, source_label)
    if args.source == "server":
        server_consistency_check(args.server_root, args.submitted, records)
    section_failures(records, manifest)
    section_metrics(records)
    if args.skip_accuracy:
        section("4. Accuracy <-> hardware join (NATS-Bench API)")
        print("  [skipped via --skip-accuracy]")
    else:
        section_accuracy(records, manifest, args.bench)
    if args.with_memory:
        section_memory(records)

    print("\nDone.")


if __name__ == "__main__":
    main()
