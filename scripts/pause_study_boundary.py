#!/usr/bin/env python3
"""Pause an old study runner only after a specified run has been fully recorded."""

import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    directory = Path(args.output)
    request = {"old_runner_pid": args.pid, "seed": args.seed, "model": args.model,
               "status": "waiting_for_run_to_finish", "purpose": "User-requested study schedule revision"}
    request_path = directory / "handoff.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    records_path = directory / "seeds" / f"seed-{args.seed}" / "training_records.json"
    while True:
        os.kill(args.pid, 0)
        if records_path.exists() and args.model in json.loads(records_path.read_text()):
            os.kill(args.pid, signal.SIGSTOP)
            request["status"] = "paused_at_completed_run_boundary"
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            print(f"Paused runner {args.pid} after {args.model}, seed {args.seed}, completed.", flush=True)
            return
        time.sleep(0.1)


if __name__ == "__main__":
    main()
