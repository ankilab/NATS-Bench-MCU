"""Submit every architecture that has no result on the MIMaaS server."""
import argparse
import csv
import glob
import json
import os
import re
from mimaas import MIMaaSClient

ARCH_RE = re.compile(r"nats_bench_model_(\d+)_int8\.tflite$")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", default="/home/ankilab/hw-nats-bench/tflite_models")
    p.add_argument("--results-dir", default="/home/ankilab/mimaas-server-test/mimaas-server/data/results")
    p.add_argument("--csv", default="submitted_requests.csv")
    p.add_argument("--device", default="nrf5340dk")
    p.add_argument("--limit", type=int, default=None, help="Max number to submit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    expected = {int(e) for e in os.listdir(args.models_dir) if e.isdigit()}

    evaluated = set()
    for meta_path in glob.glob(os.path.join(args.results_dir, "user_*", "request_*", "meta.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        m = ARCH_RE.match(meta.get("network_file_name", ""))
        if m:
            evaluated.add(int(m.group(1)))

    missing = sorted(expected - evaluated)
    if args.limit:
        missing = missing[: args.limit]
    print(f"{len(missing)} arch(s) missing on server.")

    if args.dry_run:
        for idx in missing[:20]:
            print(f"  would submit {idx:05d}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return

    client = MIMaaSClient(api_url="http://127.0.0.1:5000")
    client.login("pro_user_1", "Test1234!")

    csv_exists = os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if not csv_exists:
            w.writerow(["model", "request_id"])
        for idx in missing:
            model_path = os.path.join(args.models_dir, f"{idx:05d}",
                                      f"nats_bench_model_{idx:05d}_int8.tflite")
            if not os.path.exists(model_path):
                print(f"  no local file for {idx:05d}, skipping")
                continue
            req = client.submit_request(model_path, args.device)
            w.writerow([model_path, req.id])
            print(f"  Submitted {idx:05d} -> request ID {req.id}")


if __name__ == "__main__":
    main()
