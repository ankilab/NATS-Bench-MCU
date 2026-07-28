#!/usr/bin/env python3
"""
Apply the corrections/ overlay onto an extracted artifacts/ tree to produce the
fully corrected NATS-Bench-MCU dataset.

Why this exists
---------------
The 16 released artifacts_*.tar.gz archives are the immutable record of what was
measured on-device during the campaign. A small number of per-architecture records
were corrected after the fact (2 measurements recovered/remeasured, 115 flash-
infeasible failures relabelled to a single taxonomy). Those fixes live in this
`corrections/` overlay so the raw campaign archives stay byte-identical. This script
merges the overlay in, giving you a single corrected `artifacts/` tree.

You do NOT need this to use the tabular data: the released `export/*.csv` already
reflect every correction. Run this only if you work from the raw per-architecture
artifacts.

Overlay rule (matches analyze_artifacts.build_records(corrections_iter=...))
--------------------------------------------------------------------------
For each corrections/NNNNN/:
  * meta.json status "pending"/"excluded"  -> the architecture is removed from the
    dataset (awaiting remeasurement). (None in the current release.)
  * otherwise -> the correction is authoritative for that architecture's record.
    The merged directory contains exactly the correction's files, plus the original
    INT8 model file (nats_bench_model_*.tflite) when the correction does not ship one
    (the model is the fixed input, not part of the corrected measurement record).
Any other original file (e.g. a stale error.log for a recovered architecture) is
dropped. The operation is idempotent.

Usage
-----
    # from a directory holding both extracted trees side by side:
    python corrections/apply_corrections.py

    # or point at them explicitly:
    python corrections/apply_corrections.py --artifacts path/to/artifacts \
                                            --corrections path/to/corrections

    python corrections/apply_corrections.py --dry-run   # show what would change
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ARCH_RE = re.compile(r"^\d{5}$")
HERE = Path(__file__).resolve().parent


def load_status(meta_path: Path) -> str:
    try:
        return json.loads(meta_path.read_text()).get("status", "")
    except Exception:
        return ""


def apply_one(dst: Path, src: Path, dry_run: bool) -> str:
    """Merge one correction dir src -> artifact dir dst. Returns an action label."""
    status = load_status(src / "meta.json")
    if status in ("pending", "excluded"):
        if dst.exists() and not dry_run:
            shutil.rmtree(dst)
        return "excluded"

    corr_files = [p for p in src.iterdir() if p.is_file()]
    corr_has_tflite = any(p.suffix == ".tflite" for p in corr_files)
    orig_tflites = [p for p in dst.glob("*.tflite")] if dst.exists() else []
    preserved = (not corr_has_tflite) and bool(orig_tflites)

    if dry_run:
        return "override+keep-model" if preserved else "override"

    # Build the merged contents in a temp dir, then swap it in atomically.
    tmp = Path(tempfile.mkdtemp(prefix=".merge_", dir=dst.parent))
    try:
        for p in corr_files:
            shutil.copy2(p, tmp / p.name)
        if preserved:
            for t in orig_tflites:
                shutil.copy2(t, tmp / t.name)
        if dst.exists():
            shutil.rmtree(dst)
        tmp.replace(dst)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return "override+keep-model" if preserved else "override"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=HERE.parent / "artifacts",
                    help="Extracted artifacts/ tree to correct in place "
                         "(default: sibling of this corrections/ dir).")
    ap.add_argument("--corrections", type=Path, default=HERE,
                    help="corrections/ overlay dir (default: this script's dir).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without modifying anything.")
    ap.add_argument("--quiet", action="store_true", help="Only print the summary.")
    args = ap.parse_args()

    if not args.artifacts.is_dir():
        print(f"ERROR: artifacts dir not found: {args.artifacts}\n"
              f"Extract the artifacts_*.tar.gz archives first, or pass --artifacts.",
              file=sys.stderr)
        return 1
    if not args.corrections.is_dir():
        print(f"ERROR: corrections dir not found: {args.corrections}", file=sys.stderr)
        return 1

    arch_dirs = sorted(d for d in args.corrections.iterdir()
                       if d.is_dir() and ARCH_RE.match(d.name))
    if not arch_dirs:
        print(f"ERROR: no NNNNN/ correction dirs under {args.corrections}",
              file=sys.stderr)
        return 1

    counts = {"override": 0, "override+keep-model": 0, "excluded": 0, "created": 0}
    for src in arch_dirs:
        dst = args.artifacts / src.name
        existed = dst.exists()
        action = apply_one(dst, src, args.dry_run)
        counts[action] += 1
        if not existed and action != "excluded":
            counts["created"] += 1
        if not args.quiet:
            tag = "would " if args.dry_run else ""
            note = "" if existed else "  (arch not in extracted tree; created)"
            print(f"  {tag}{action:20s} {src.name}{note}")

    verb = "Would apply" if args.dry_run else "Applied"
    total = counts["override"] + counts["override+keep-model"]
    print(f"\n{verb} {total} correction(s) "
          f"({counts['override+keep-model']} kept the original model file), "
          f"{counts['excluded']} architecture(s) removed.")
    if counts["created"]:
        print(f"Note: {counts['created']} architecture(s) were not present in the "
              f"artifacts tree and were created from the overlay.")
    if not args.dry_run:
        print(f"Corrected dataset ready at: {args.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
