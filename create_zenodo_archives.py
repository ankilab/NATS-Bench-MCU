"""
Create chunked tar.gz archives of the artifacts/ directory for Zenodo upload.

Each archive contains CHUNK_SIZE consecutive architecture directories.
An index.json is written alongside the archives so that dataset_example.py
can locate which archive contains a given architecture index without
downloading anything other than the index.

Usage
-----
    python create_zenodo_archives.py --out zenodo_upload/

Optional arguments
------------------
    --artifacts   Path to artifacts/ directory (default: ./artifacts/)
    --chunk-size  Number of architectures per archive (default: 1000)
    --out         Output directory for archives and index.json (default: ./zenodo_upload/)

Output
------
    zenodo_upload/
    ├── index.json                         Maps arch_idx → archive filename
    ├── artifacts_00000-00999.tar.gz
    ├── artifacts_01000-01999.tar.gz
    ├── ...
    └── artifacts_15000-15624.tar.gz
"""

import argparse
import json
import sys
import tarfile
from pathlib import Path


TOTAL_ARCHS = 15_625
DEFAULT_CHUNK = 1_000


def build_chunks(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) pairs."""
    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        chunks.append((start, end))
        start = end
    return chunks


def archive_name(start: int, end: int) -> str:
    return f"artifacts_{start:05d}-{end - 1:05d}.tar.gz"


def create_archives(artifacts_root: Path, out_dir: Path, chunk_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(TOTAL_ARCHS, chunk_size)

    index: dict[str, str] = {}  # arch_idx (str) → archive filename

    for chunk_idx, (start, end) in enumerate(chunks):
        name = archive_name(start, end)
        out_path = out_dir / name
        print(f"[{chunk_idx + 1}/{len(chunks)}] Creating {name}  (archs {start:05d}–{end - 1:05d})")

        if out_path.exists():
            print(f"  Already exists — skipping. Delete to recreate.")
            # Still populate index from existing archive
            for i in range(start, end):
                index[str(i)] = name
            continue

        with tarfile.open(out_path, "w:gz", compresslevel=6) as tf:
            for i in range(start, end):
                arch_dir = artifacts_root / f"{i:05d}"
                if not arch_dir.exists():
                    print(f"  WARNING: {arch_dir} not found — skipping.")
                    continue
                tf.add(arch_dir, arcname=f"{i:05d}", recursive=True)
                index[str(i)] = name

        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Done  ({size_mb:.0f} MB)")

    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, separators=(",", ":"))
    print(f"\nIndex written to {index_path}  ({len(index)} entries)")
    print(f"\nZenodo upload directory: {out_dir.resolve()}")
    print(f"Files to upload: {len(list(out_dir.iterdir()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"),
                        help="Path to the artifacts/ directory (default: ./artifacts/)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK,
                        help=f"Architectures per archive (default: {DEFAULT_CHUNK})")
    parser.add_argument("--out", type=Path, default=Path("zenodo_upload"),
                        help="Output directory (default: ./zenodo_upload/)")
    args = parser.parse_args()

    if not args.artifacts.exists():
        print(f"ERROR: artifacts directory not found: {args.artifacts}")
        sys.exit(1)

    chunks = build_chunks(TOTAL_ARCHS, args.chunk_size)
    print(f"Artifacts:  {args.artifacts.resolve()}")
    print(f"Output:     {args.out.resolve()}")
    print(f"Chunk size: {args.chunk_size} architectures → {len(chunks)} archives")
    print()

    create_archives(args.artifacts, args.out, args.chunk_size)


if __name__ == "__main__":
    main()
