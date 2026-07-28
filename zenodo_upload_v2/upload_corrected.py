"""
Upload ONLY the camera-ready corrected files to a NEW VERSION of the Zenodo record.

The 16 large artifacts_*.tar.gz archives are unchanged and are inherited
automatically when you create the new version on Zenodo, so they are NOT re-uploaded
here. This script uploads only:

    export.tar.gz        (corrected export CSVs; same filename -> replaces the old one)
    corrections.tar.gz   (new overlay: recovered 07650 + remeasured 07651 + 115 relabels)

Workflow
--------
1. On the Zenodo record page, log in and click "New version". This creates a draft
   that inherits all existing files. Note the DRAFT's deposition id (the number in
   the draft URL, e.g. https://zenodo.org/uploads/<ID> or /deposit/<ID>).
2. Run:

       export ZENODO_TOKEN=<your token>
       python zenodo_upload_v2/upload_corrected.py <DRAFT_DEPOSITION_ID>

3. Back on Zenodo: confirm export.tar.gz was replaced and corrections.tar.gz added,
   fill in the version notes (see VERSION_NOTES.txt), then Publish.

Uploading export.tar.gz under its existing filename replaces the inherited copy in
the draft's bucket; corrections.tar.gz is added as a new file.
"""

import subprocess
import sys
from pathlib import Path

UPLOAD_SCRIPT = Path("/home/ankilab/zenodo-upload/zenodo_upload.sh")
UPLOAD_DIR = Path(__file__).parent
FILES = ["export.tar.gz", "corrections.tar.gz"]

if len(sys.argv) != 2:
    print("usage: python upload_corrected.py <NEW_VERSION_DEPOSITION_ID>")
    print("       (ZENODO_TOKEN must be set in the environment)")
    sys.exit(2)

deposition_id = sys.argv[1]

if not UPLOAD_SCRIPT.exists():
    print(f"ERROR: upload script not found: {UPLOAD_SCRIPT}")
    sys.exit(1)

for name in FILES:
    f = UPLOAD_DIR / name
    if not f.is_file():
        print(f"ERROR: missing {f}")
        sys.exit(1)
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"Uploading {name} ({size_mb:.1f} MB) to deposition {deposition_id}")
    result = subprocess.run(
        ["bash", str(UPLOAD_SCRIPT), deposition_id, str(f), "-v"],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: upload failed for {name} (exit {result.returncode})")
        sys.exit(result.returncode)
    print()

print("Done. Now set the version notes and Publish on Zenodo.")
