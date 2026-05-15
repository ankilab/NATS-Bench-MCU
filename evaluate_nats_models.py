import argparse
import csv
import glob
import json
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from mimaas import MIMaaSClient
from mimaas.exceptions import ResourceNotFoundError

client = MIMaaSClient(api_url="http://127.0.0.1:5000")


def register_and_login():
    # Register a new user (only needed once)
    client.register(
        username="pro_user_1",
        email="prouser1@example.com",
        first_name="Sebi",
        surname="Doe3",
        password="Test1234!",
        invite_token="0efd04204930f74f3a745c3b0882b8b31143e129059035203683025105e37b18",
        plan="admin"
    )
    client.login("pro_user_1", "Test1234!")


def login():
    # Login with existing user
    client.login("pro_user_1", "Test1234!")


def submit_model(model_path, device):
    # Submit a model for evaluation
    request = client.submit_request(model_path, device)
    return request.id


def get_results(request_id):
    # Get results for a specific request
    results = client.get_results(request_id)
    return results


def download_artifacts(request_id, destination_folder):
    # Download all artifacts for a specific request
    client.download_all_artifacts(request_id, destination_folder)


def cmd_register(args):
    register_and_login()
    print("Registered and logged in.")


def _load_submitted(csv_path):
    """Return set of already-submitted absolute model paths from the CSV."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r") as f:
        return {row["model"] for row in csv.DictReader(f)}


def cmd_submit(args):
    login()
    all_models = sorted(glob.glob(os.path.join(args.models_dir, "**/*.tflite"), recursive=True))
    if args.quantized:
        all_models = [m for m in all_models if m.endswith("_int8.tflite")]
    else:
        all_models = [m for m in all_models if not m.endswith("_int8.tflite")]

    total = len(all_models)
    end = min(args.end, total) if args.end is not None else total
    # Select only the models in the specified range
    all_models = all_models[args.start:end]
    if args.start or args.end is not None:
        print(f"Considering models [{args.start}:{end}) of {total}.")

    already_submitted = _load_submitted(args.csv)
    models = [m for m in all_models if os.path.abspath(m) not in {os.path.abspath(p) for p in already_submitted}]

    skipped = len(all_models) - len(models)
    if skipped:
        print(f"Skipping {skipped} already-submitted model(s).")
    if not models:
        print("Nothing new to submit.")
        return

    print(f"Submitting {len(models)} model(s).")
    csv_exists = os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["model", "request_id"])
        for model in models:
            request_id = submit_model(model, args.device)
            writer.writerow([model, request_id])
            print(f"  Submitted {os.path.basename(model)} -> request ID {request_id}")
    print(f"Done. {len(models)} new submission(s) appended to {args.csv}.")


_ARCH_RE = re.compile(r"nats_bench_model_(\d+)(_int8)?\.tflite$")


def _is_pipeline_exception(request_dir):
    error_log = os.path.join(request_dir, "error.log")
    if not os.path.exists(error_log):
        return False
    try:
        with open(error_log) as f:
            return "pipeline_exception" in f.read()
    except OSError:
        return False


def cmd_verify(args):
    expected = set()
    for entry in os.listdir(args.models_dir):
        try:
            expected.add(int(entry))
        except ValueError:
            continue

    # Collect the most recent request per arch (highest request_id wins)
    arch_best = {}  # arch_idx -> {rid, status, failed_stage, request_dir}
    scanned = 0
    for meta_path in glob.glob(os.path.join(args.results_dir, "user_*", "request_*", "meta.json")):
        scanned += 1
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        fname = meta.get("network_file_name", "")
        m = _ARCH_RE.match(fname)
        if not m:
            continue
        if (m.group(2) is not None) != args.quantized:
            continue
        idx = int(m.group(1))
        rid = meta.get("request_id") or 0
        prev = arch_best.get(idx)
        if prev is None or rid > prev["rid"]:
            arch_best[idx] = {
                "rid": rid,
                "status": meta.get("status", "unknown"),
                "failed_stage": meta.get("failed_stage"),
                "request_dir": os.path.dirname(meta_path),
            }

    # Categorize each arch by its most recent result
    evaluated = {}    # arch_idx -> status
    broken = {}       # arch_idx -> request_id  (pipeline_exception)
    flash_failed = {} # arch_idx -> request_id  (flash_failed internal error)
    for idx, best in arch_best.items():
        status, failed_stage = best["status"], best["failed_stage"]
        if status not in ("done", "error") and _is_pipeline_exception(best["request_dir"]):
            broken[idx] = best["rid"]
        elif status == "error" and failed_stage == "flash_failed":
            flash_failed[idx] = best["rid"]
        else:
            evaluated[idx] = status

    need_resubmit = sorted((expected - evaluated.keys()) | set(broken) | set(flash_failed))
    done_count = sum(1 for s in evaluated.values() if s == "done")
    error_count = sum(1 for s in evaluated.values() if s == "error")
    other_count = len(evaluated) - done_count - error_count

    kind = "quantized" if args.quantized else "float"
    print(f"Scanned {scanned} request folder(s) under {args.results_dir}.")
    print(f"Expected ({kind}): {len(expected)} arch(s).")
    print(f"Evaluated: {len(evaluated)} ({done_count} done, {error_count} error, {other_count} other).")
    if broken:
        print(f"Broken (pipeline_exception): {len(broken)} — need resubmission.")
    if flash_failed:
        print(f"Broken (flash_failed): {len(flash_failed)} — need resubmission.")

    missing_from_server = sorted(expected - evaluated.keys() - set(broken) - set(flash_failed))
    if missing_from_server:
        print(f"Missing from server: {len(missing_from_server)}")
        for idx in missing_from_server[:50]:
            print(f"  {idx:05d}")
        if len(missing_from_server) > 50:
            print(f"  ... and {len(missing_from_server) - 50} more")

    if not need_resubmit:
        print("All architectures have valid results.")

    if args.output_missing and need_resubmit:
        with open(args.output_missing, "w") as f:
            for idx in need_resubmit:
                f.write(f"{idx:05d}\n")
        print(f"Wrote {len(need_resubmit)} indices needing resubmission to {args.output_missing}.")

    if args.resubmit_broken and broken:
        _resubmit_broken(args, broken)
    if args.resubmit_flash_failed and flash_failed:
        _resubmit_broken(args, flash_failed)


def _resubmit_broken(args, broken):
    """Re-submit broken arches and rewrite the CSV replacing old stale rows."""
    login()
    csv_rows = []
    if os.path.exists(args.csv):
        with open(args.csv) as f:
            csv_rows = list(csv.DictReader(f))

    stale_request_ids = {str(rid) for rid in broken.values() if rid is not None}
    clean_rows = [r for r in csv_rows if r["request_id"] not in stale_request_ids]

    print(f"\nRe-submitting {len(broken)} broken arch(s)...")
    new_rows = []
    for idx in sorted(broken):
        model_path = os.path.join(args.models_dir, f"{idx:05d}",
                                  f"nats_bench_model_{idx:05d}_int8.tflite")
        if not os.path.exists(model_path):
            print(f"  no local file for {idx:05d}, skipping")
            continue
        req = client.submit_request(model_path, args.device)
        new_rows.append({"model": model_path, "request_id": str(req.id)})
        print(f"  Submitted {idx:05d} -> request ID {req.id}")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "request_id"])
        w.writeheader()
        w.writerows(clean_rows + new_rows)
    print(f"CSV updated: {len(new_rows)} resubmitted, {len(stale_request_ids)} stale row(s) replaced.")


def cmd_results(args):
    login()
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results = get_results(row["request_id"])
            print(f"{row['model']}: {results}")


# Files required for a complete successful result.
# The server has two formats:
#   - newer: logs stored as .gz, power samples as .parquet
#   - older: uncompressed logs, power samples as .zip
# _artifacts_complete accepts either format.
_SUCCESS_REQUIRED = {"meta.json", "results.json", "ppk2_summary.csv", "uart.log"}
_SUCCESS_COMPRESSED = {"flash.log.gz", "model.cpp.gz", "ram.json.gz", "rom.json.gz"}
_SUCCESS_UNCOMPRESSED = {"flash.log", "model.cpp", "ram.json", "rom.json"}
# Keep for external callers that reference SUCCESS_ARTIFACT_FILES directly
SUCCESS_ARTIFACT_FILES = _SUCCESS_REQUIRED | _SUCCESS_UNCOMPRESSED | {"ppk2_samples.zip"}

# A failed flash produces only an error log + meta + the original tflite.
# That's a terminal state on the server — no more data will arrive — so treat
# it as "complete, just failed" and stop re-downloading.
FAILURE_ARTIFACT_FILES = {"error.log", "meta.json"}


def _scan_complete_models(output_dir):
    """Return set of model_names already fully downloaded in output_dir."""
    complete = set()
    if not os.path.isdir(output_dir):
        return complete
    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue
        try:
            present = set(os.listdir(entry.path))
        except OSError:
            continue
        tflite = next((f for f in present if f.endswith(".tflite")), None)
        if tflite is None:
            continue
        has_required = _SUCCESS_REQUIRED | {tflite} <= present
        has_power = "ppk2_samples.zip" in present or "ppk2_samples.parquet" in present
        logs_ok = _SUCCESS_COMPRESSED <= present or _SUCCESS_UNCOMPRESSED <= present
        if (has_required and has_power and logs_ok) or (FAILURE_ARTIFACT_FILES | {tflite} <= present):
            complete.add(entry.name)
    return complete


def _download_one(request_id, model_name, output_dir):
    """Download and extract one artifact. Returns (model_name, status, message)."""
    zip_path = os.path.join(output_dir, f"{model_name}.zip")
    extract_dir = os.path.join(output_dir, model_name)
    try:
        client.download_all_artifacts(request_id, zip_path)
    except ResourceNotFoundError as e:
        return (model_name, "not_ready", str(e))
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    os.remove(zip_path)
    return (model_name, "ok", "")


def cmd_download(args):
    login()
    os.makedirs(args.output, exist_ok=True)

    # Scan output dir once upfront instead of per-row os.listdir calls
    print("Scanning already-downloaded artifacts...", flush=True)
    complete_set = _scan_complete_models(args.output)
    print(f"  {len(complete_set)} already complete.", flush=True)

    # Read CSV and deduplicate: keep only the most recent request_id per model_name
    pending = {}  # model_name -> (request_id, model_path)
    incomplete = []
    with open(args.csv, "r") as f:
        for row in csv.DictReader(f):
            model_path = row["model"]
            model_name = os.path.basename(os.path.dirname(model_path))
            if model_name in complete_set:
                continue
            extract_dir = os.path.join(args.output, model_name)
            if os.path.isdir(extract_dir):
                present = set(os.listdir(extract_dir))
                missing = (_SUCCESS_REQUIRED | _SUCCESS_UNCOMPRESSED | {os.path.basename(model_path)}) - present
                if missing and model_name not in incomplete:
                    print(f"Incomplete: {model_name} (missing: {', '.join(sorted(missing))}), re-downloading.")
                    incomplete.append(model_name)
            # Keep highest request_id per model_name (most recent)
            rid = int(row["request_id"])
            if model_name not in pending or rid > pending[model_name][0]:
                pending[model_name] = (rid, model_path)

    skipped = len(complete_set)
    if not pending:
        print(f"Nothing to download. {skipped} already complete.")
        return
    print(f"{len(pending)} model(s) to download ({len(incomplete)} incomplete re-downloads).")

    downloaded = 0
    not_ready = 0
    print_lock = Lock()

    def task(item):
        model_name, (request_id, _) = item
        return _download_one(request_id, model_name, args.output)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, item): item[0] for item in pending.items()}
        for future in as_completed(futures):
            model_name, status, msg = future.result()
            with print_lock:
                if status == "ok":
                    print(f"  Downloaded {model_name}")
                elif status == "not_ready":
                    print(f"  Not ready: {model_name} ({msg})")

    # Recount from disk for accurate totals
    final_complete = _scan_complete_models(args.output)
    downloaded = len(final_complete) - skipped
    not_ready = len(pending) - downloaded
    print(f"Done: {downloaded} downloaded, {skipped} already complete, {not_ready} not ready.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NATS-Bench TFLite models via MIMaaS")
    sub = parser.add_subparsers(dest="command", required=True)

    # register
    sub.add_parser("register", help="Register a new user and login")

    # submit
    p_submit = sub.add_parser("submit", help="Submit all TFLite models for evaluation")
    p_submit.add_argument("--models-dir", default="/home/ankilab/hw-nats-bench/tflite_models",
                          help="Directory containing TFLite models")
    p_submit.add_argument("--device", default="nrf5340dk", help="Target device")
    p_submit.add_argument("--csv", default="submitted_requests.csv",
                          help="CSV file to save request IDs")
    p_submit.add_argument("--quantized", action=argparse.BooleanOptionalAction, default=True,
                          help="Submit INT8 quantized models (default) or float32 with --no-quantized")
    p_submit.add_argument("--start", type=int, default=0,
                          help="Start index into the sorted model list (inclusive)")
    p_submit.add_argument("--end", type=int, default=None,
                          help="End index into the sorted model list (exclusive, default: all)")

    # verify
    p_verify = sub.add_parser("verify",
                              help="Check that every local architecture has a result on the server")
    p_verify.add_argument("--models-dir", default="/home/ankilab/hw-nats-bench/tflite_models",
                          help="Directory containing TFLite model folders (one per arch index)")
    p_verify.add_argument("--results-dir",
                          default="/home/ankilab/mimaas-server-test/mimaas-server/data/results",
                          help="Server-side results directory to scan")
    p_verify.add_argument("--quantized", action=argparse.BooleanOptionalAction, default=True,
                          help="Verify INT8 quantized models (default) or float32 with --no-quantized")
    p_verify.add_argument("--output-missing", default=None,
                          help="If set, write the full list of missing arch indices to this file")
    p_verify.add_argument("--resubmit-broken", action="store_true",
                          help="Re-submit broken (pipeline_exception) arches and update the CSV")
    p_verify.add_argument("--resubmit-flash-failed", action="store_true",
                          help="Re-submit flash_failed arches and update the CSV")
    p_verify.add_argument("--device", default="nrf5340dk", help="Target device (used with --resubmit-broken)")
    p_verify.add_argument("--csv", default="submitted_requests.csv",
                          help="CSV to update when using --resubmit-broken")

    # results
    p_results = sub.add_parser("results", help="Print results for submitted models")
    p_results.add_argument("--csv", default="submitted_requests.csv",
                           help="CSV file with request IDs")

    # download
    p_download = sub.add_parser("download", help="Download all artifacts for submitted models")
    p_download.add_argument("--csv", default="submitted_requests.csv",
                            help="CSV file with request IDs")
    p_download.add_argument("--output", default="artifacts",
                            help="Output directory for downloaded artifacts")
    p_download.add_argument("--workers", type=int, default=8,
                            help="Number of parallel download threads (default: 8)")

    args = parser.parse_args()

    commands = {
        "register": cmd_register,
        "submit": cmd_submit,
        "verify": cmd_verify,
        "results": cmd_results,
        "download": cmd_download,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

