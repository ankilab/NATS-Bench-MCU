"""
Upload all files in this directory to Zenodo using zenodo_upload.sh.

Usage
-----
    python zenodo_upload/upload_all.py

Environment
-----------
    ZENODO_TOKEN   Zenodo personal access token (required by zenodo_upload.sh)

The deposition ID and upload script path are hardcoded below — adjust if needed.
"""

import subprocess
import sys
from pathlib import Path

DEPOSITION_ID = "20204556"
UPLOAD_SCRIPT = Path("/home/ankilab/zenodo-upload/zenodo_upload.sh")
UPLOAD_DIR = Path(__file__).parent

if not UPLOAD_SCRIPT.exists():
    print(f"ERROR: upload script not found: {UPLOAD_SCRIPT}")
    sys.exit(1)

files = sorted(UPLOAD_DIR.iterdir())
files = [f for f in files if f.is_file() and f.name != Path(__file__).name]

print(f"Uploading {len(files)} file(s) to Zenodo deposition {DEPOSITION_ID}\n")

for i, f in enumerate(files, 1):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"[{i}/{len(files)}] {f.name}  ({size_mb:.0f} MB)")
    result = subprocess.run(
        ["bash", str(UPLOAD_SCRIPT), DEPOSITION_ID, str(f), "-v"],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: upload failed for {f.name} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print()

print("All files uploaded successfully.")
