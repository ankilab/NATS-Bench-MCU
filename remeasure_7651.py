#!/usr/bin/env python3
"""
Remeasure architecture 07651 on the nRF5340-DK via MIMaaS.

Background
----------
During the original campaign, arch 07651 (INT8 flatbuffer ~363 KB, well within the
746 KB deployable ceiling) failed for infrastructure reasons only: its May-4 build
started flashing but the pipeline died mid-flash (GPIO-timeout / orphaned-process bug,
since fixed). It carries no device measurements. Under the corrected precheck it passes
cleanly, so it should be deployed and profiled like any other feasible architecture.

This script submits the existing INT8 flatbuffer to the (fixed) MIMaaS server, waits for
the run to finish, and downloads the artifacts into corrections/07651/ so the export
pipeline can pick them up as an overlay on the released dataset.

Prerequisites
-------------
  * MIMaaS server running locally (default http://127.0.0.1:5000) with the fixed
    validation.py and board configs.
  * nRF5340-DK + Power Profiler Kit II connected and powered.

Usage
-----
    python remeasure_7651.py                 # defaults are fine
    python remeasure_7651.py --board nrf5340dk_1 --timeout 2400
"""
import argparse
import os
import zipfile

from mimaas import MIMaaSClient
from mimaas.exceptions import ProcessingError, ResourceNotFoundError

# Same account the campaign used (see evaluate_nats_models.py).
USERNAME = "pro_user_1"
PASSWORD = "Test1234!"

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(
    REPO, "tflite_models", "07651", "nats_bench_model_07651_int8.tflite"
)
DEFAULT_OUTPUT = os.path.join(REPO, "corrections", "07651")


def summarize(extract_dir):
    """Print a short human-readable summary of the downloaded measurement."""
    ppk2 = os.path.join(extract_dir, "ppk2_summary.csv")
    uart = os.path.join(extract_dir, "uart.log")
    if os.path.exists(uart):
        n_ok = sum(1 for l in open(uart) if "inference completed successfully" in l)
        print(f"  inferences completed successfully: {n_ok}")
    if os.path.exists(ppk2):
        import csv as _csv
        rows = list(_csv.DictReader(open(ppk2)))
        if rows:
            def avg(k):
                vals = [float(r[k]) for r in rows if r.get(k)]
                return sum(vals) / len(vals) if vals else float("nan")
            print(f"  power segments: {len(rows)}")
            print(f"  mean duration : {avg('duration_s'):.4f} s")
            print(f"  mean current  : {avg('avg_current_uA'):.1f} uA")
            print(f"  mean energy   : {avg('energy_uJ') / 1000:.4f} mJ")
    print(f"\nArtifacts saved to: {extract_dir}")
    print("Files:", ", ".join(sorted(os.listdir(extract_dir))))


def main():
    ap = argparse.ArgumentParser(description="Remeasure arch 07651 on the nRF5340-DK")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="INT8 TFLite flatbuffer to submit")
    ap.add_argument("--board", default="nrf5340dk", help="Target board (default: nrf5340dk)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Directory for downloaded artifacts")
    ap.add_argument("--api-url", default="http://127.0.0.1:5000", help="MIMaaS server URL")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Max seconds to wait for build+flash+measure (default: 1800)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    client = MIMaaSClient(api_url=args.api_url)
    try:
        client.login(USERNAME, PASSWORD)
    except Exception as e:
        raise SystemExit(
            f"Login failed ({e}).\nIs the MIMaaS server running at {args.api_url}?"
        )

    print(f"Submitting {os.path.basename(args.model)} to board '{args.board}' ...")
    req = client.submit_request(args.model, args.board, quantize=False)
    print(f"  request ID: {req.id}")
    print(f"Waiting up to {args.timeout}s for the run to finish (polling every 5s) ...")

    os.makedirs(args.output, exist_ok=True)
    try:
        results = client.wait_for_completion(req.id, timeout=args.timeout, poll_interval=5)
        print(f"\n✓ Run completed. Server results: {results}")
    except ProcessingError as e:
        print(f"\n✗ Run FAILED on the server: {e}")
        try:
            client.download_error_log(req.id, os.path.join(args.output, "error.log"))
            print(f"  error log saved to {os.path.join(args.output, 'error.log')}")
        except ResourceNotFoundError:
            pass
        raise SystemExit(1)

    # Download the full artifact bundle and extract it into the output dir.
    zip_path = os.path.join(args.output, "07651.zip")
    client.download_all_artifacts(req.id, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.output)
    os.remove(zip_path)

    print("\n=== Measurement summary (arch 07651) ===")
    summarize(args.output)


if __name__ == "__main__":
    main()
