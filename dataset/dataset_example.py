"""
NATS-Bench-MCU dataset example
===============================
Demonstrates how to read and export raw measurement artifacts for a single
architecture index from the chunked Zenodo archives.

The dataset is distributed as a set of tar.gz archives (one per ~1,000
architectures) plus an index.json that maps each architecture index to the
archive that contains it.  This script reads only the relevant archive and
extracts only the requested directory — it never loads the full dataset.

Dataset layout (after downloading from Zenodo)
----------------------------------------------
    dataset/
    ├── index.json                     Maps arch_idx → archive filename
    ├── artifacts_00000-00999.tar.gz
    ├── artifacts_01000-01999.tar.gz
    ├── ...
    ├── artifacts_15000-15624.tar.gz
    ├── export/
    │   ├── hw_metrics.csv
    │   ├── accuracies.csv
    │   ├── failures.csv
    │   ├── coverage.csv
    │   └── README.md
    ├── DATASET_README.md
    └── dataset_example.py             ← this file

Usage
-----
Print a summary of one evaluation to stdout:
    python dataset_example.py print 42

Export all files for one evaluation to a decompressed folder:
    python dataset_example.py export 42 --out ./evaluation_42/

Pass --dataset if the dataset root is not the directory containing this script:
    python dataset_example.py print 42 --dataset /path/to/dataset/
"""

import argparse
import csv
import gzip
import io
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Archive / index helpers
# ---------------------------------------------------------------------------

def _load_index(dataset_root: Path) -> dict[str, str]:
    index_path = dataset_root / "index.json"
    if not index_path.exists():
        print(f"ERROR: index.json not found in {dataset_root}")
        print("Make sure you are pointing --dataset at the Zenodo download folder.")
        sys.exit(1)
    with open(index_path) as f:
        return json.load(f)


def _find_archive(dataset_root: Path, arch_idx: int, index: dict) -> Path:
    archive_name = index.get(str(arch_idx))
    if archive_name is None:
        print(f"ERROR: architecture index {arch_idx} not found in index.json")
        sys.exit(1)
    archive_path = dataset_root / archive_name
    if not archive_path.exists():
        print(f"ERROR: archive not found: {archive_path}")
        sys.exit(1)
    return archive_path


def _extract_arch_to_tmpdir(archive_path: Path, arch_idx: int) -> Path:
    """Extract one architecture directory from a tar.gz into a temp directory."""
    prefix = f"{arch_idx:05d}/"
    tmp = Path(tempfile.mkdtemp(prefix="natsbench_"))
    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.name == f"{arch_idx:05d}" or m.name.startswith(prefix)]
        if not members:
            print(f"ERROR: architecture {arch_idx:05d} not found inside {archive_path.name}")
            shutil.rmtree(tmp, ignore_errors=True)
            sys.exit(1)
        tf.extractall(path=tmp, members=members)
    return tmp / f"{arch_idx:05d}"


# ---------------------------------------------------------------------------
# File-format helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_gz_json(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _decompress_gz(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def _parquet_to_csv(src: Path, dst: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required to convert ppk2_samples.parquet to CSV.\n"
            "Install it with:  pip install pyarrow"
        )
    table = pq.read_table(src)
    with open(dst, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(table.column_names)
        for batch in table.to_batches():
            rows = zip(*[col.to_pylist() for col in batch.columns])
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_evaluation(arch_idx: int, dataset_root: Path) -> None:
    """
    Print a human-readable summary of one architecture evaluation.

    Reads only the single tar.gz archive that contains arch_idx.
    """
    index = _load_index(dataset_root)
    archive_path = _find_archive(dataset_root, arch_idx, index)

    print(f"=== Architecture {arch_idx:05d} ===")
    print(f"Archive: {archive_path.name}\n")

    arch_dir = _extract_arch_to_tmpdir(archive_path, arch_idx)
    try:
        _print_from_dir(arch_idx, arch_dir)
    finally:
        shutil.rmtree(arch_dir.parent, ignore_errors=True)


def _print_from_dir(arch_idx: int, arch_dir: Path) -> None:
    meta = _read_json(arch_dir / "meta.json")
    print("[meta.json]")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print()

    status = meta.get("status", "unknown")

    if status == "done":
        results = _read_json(arch_dir / "results.json")
        print("[results.json]")
        print(f"  ROM usage:         {results['rom_usage']:,} bytes  ({results['rom_usage']/1024:.1f} KB)")
        print(f"  RAM usage:         {results['ram_usage']:,} bytes  ({results['ram_usage']/1024:.1f} KB)")
        print(f"  Inference latency: {results['duration_avg_s']:.4f} s")
        print(f"  Mean power:        {results['avg_power_uW']/1000:.3f} mW")
        print(f"  Energy/inference:  {results['avg_energy_uJ']/1000:.3f} mJ")
        print()

        ppk2_path = arch_dir / "ppk2_summary.csv"
        if ppk2_path.exists():
            with open(ppk2_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"[ppk2_summary.csv]  ({len(rows)} segments)")
            if rows:
                headers = list(rows[0].keys())
                print("  " + "  ".join(f"{h:>18}" for h in headers))
                for row in rows:
                    print("  " + "  ".join(f"{row[h]:>18}" for h in headers))
            print()

        uart_path = arch_dir / "uart.log"
        if uart_path.exists():
            print("[uart.log]")
            with open(uart_path) as f:
                for line in f:
                    line = line.rstrip()
                    if line:
                        print(f"  {line}")
            print()

        for label, fname in [("ROM", "rom.json.gz"), ("RAM", "ram.json.gz")]:
            gz_path = arch_dir / fname
            if gz_path.exists():
                data = _read_gz_json(gz_path)
                total_kb = data.get("total_size", 0) / 1024
                print(f"[{fname}]  total {label}: {total_kb:.1f} KB")

    else:
        error_path = arch_dir / "error.log"
        if error_path.exists():
            print("[error.log]")
            with open(error_path) as f:
                for line in f:
                    print(f"  {line}", end="")
        print()


def export_evaluation(arch_idx: int, dataset_root: Path, out_dir: Path) -> Path:
    """
    Export a fully decompressed copy of one architecture's artifacts.

    Reads only the single tar.gz archive that contains arch_idx, extracts
    the relevant directory to a temporary location, then decompresses:

        flash.log.gz  → flash.log
        model.cpp.gz  → model.cpp
        rom.json.gz   → rom.json
        ram.json.gz   → ram.json
        ppk2_samples.parquet → ppk2_samples.csv  (requires pyarrow)

    All other files are copied as-is.
    Returns the path to the exported directory.
    """
    index = _load_index(dataset_root)
    archive_path = _find_archive(dataset_root, arch_idx, index)

    print(f"Reading {archive_path.name} ...")
    arch_dir = _extract_arch_to_tmpdir(archive_path, arch_idx)

    dest = out_dir / f"{arch_idx:05d}"
    dest.mkdir(parents=True, exist_ok=True)

    GZ_FILES = {
        "flash.log.gz": "flash.log",
        "model.cpp.gz": "model.cpp",
        "rom.json.gz":  "rom.json",
        "ram.json.gz":  "ram.json",
    }

    try:
        for src_file in sorted(arch_dir.iterdir()):
            name = src_file.name

            if name in GZ_FILES:
                target_name = GZ_FILES[name]
                print(f"  decompress  {name} → {target_name}")
                _decompress_gz(src_file, dest / target_name)

            elif name == "ppk2_samples.parquet":
                target_name = "ppk2_samples.csv"
                print(f"  convert     {name} → {target_name}")
                try:
                    _parquet_to_csv(src_file, dest / target_name)
                except ImportError as e:
                    print(f"  WARNING: {e}")
                    print(f"  Copying {name} as-is instead.")
                    shutil.copy2(src_file, dest / name)

            else:
                print(f"  copy        {name}")
                shutil.copy2(src_file, dest / name)
    finally:
        shutil.rmtree(arch_dir.parent, ignore_errors=True)

    print(f"\nExported to: {dest}")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_dataset_root(script_path: Path) -> Path:
    return script_path.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NATS-Bench-MCU dataset example: inspect or export one evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the dataset root containing index.json and the .tar.gz archives "
             "(default: directory containing this script)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_print = sub.add_parser("print", help="Print a summary of one evaluation to stdout")
    p_print.add_argument("index", type=int, help="Architecture index (0–15624)")
    p_print.add_argument("--dataset", type=Path, default=None,
                         help="Dataset root (overrides top-level --dataset)")

    p_export = sub.add_parser(
        "export",
        help="Export one evaluation as decompressed files into a new folder",
    )
    p_export.add_argument("index", type=int, help="Architecture index (0–15624)")
    p_export.add_argument(
        "--out",
        type=Path,
        default=Path("./exported_evaluation"),
        help="Output directory (default: ./exported_evaluation/)",
    )
    p_export.add_argument("--dataset", type=Path, default=None,
                          help="Dataset root (overrides top-level --dataset)")

    args = parser.parse_args()

    # --dataset can be given either before or after the subcommand
    sub_dataset = getattr(args, "dataset", None)
    top_dataset = parser.parse_known_args()[0].dataset
    dataset_root = sub_dataset or top_dataset or _default_dataset_root(Path(__file__))

    if args.command == "print":
        print_evaluation(args.index, dataset_root)
    elif args.command == "export":
        export_evaluation(args.index, dataset_root, args.out)


if __name__ == "__main__":
    main()
