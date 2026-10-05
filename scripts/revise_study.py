#!/usr/bin/env python3
"""Apply the user-requested two-seed schedule after the active run finishes."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from trafficbench.discovery import sha256
from trafficbench.study import implementation_hashes, read_records, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.output)
    handoff = json.loads((directory / "handoff.json").read_text())
    if handoff["status"] != "paused_at_completed_run_boundary":
        raise ValueError("The active seed 21 regimeliquid run must finish before migration")
    records = read_records(directory, [7, 21, 42, 87, 123])
    if "regimeliquid" not in records["21"]:
        raise ValueError("Active run was not recorded")
    plan = json.loads((directory / "plan.json").read_text())
    if (directory / "selection.json").exists():
        raise ValueError("Cannot revise seed roles after architecture selection")
    current = implementation_hashes()
    if any(current[name] != digest for name, digest in plan["implementation_hashes"].items() if name != "study.py"):
        raise ValueError("Training/model implementation changed; cannot reuse old training runs")
    for seed, models in records.items():
        for name, record in models.items():
            if sha256(directory / "seeds" / f"seed-{seed}" / f"{name}.pt") != record["checkpoint_sha256"]:
                raise ValueError("A completed checkpoint changed")
    # The paused runner has already atomically saved the active run. Terminate it
    # before it starts the next fit, then acquire its released study lock.
    os.kill(handoff["old_runner_pid"], signal.SIGTERM)
    os.kill(handoff["old_runner_pid"], signal.SIGCONT)
    with (directory / ".study.lock").open("a") as lock:
        deadline = time.monotonic() + 30
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError("Old runner did not release its lock")
                time.sleep(0.1)
        original = directory / "plan.original.json"
        if original.exists():
            raise ValueError("Plan was already revised")
        write_json(original, plan)
        selection_seeds, confirmation_seeds = [7, 21], []
        schedule = [{"seed": seed, "model": name} for seed in selection_seeds
                    for name in [*plan["config"]["baselines"], *plan["config"]["candidates"]]]
        plan["config"]["seeds"] = selection_seeds
        plan["config"]["deployment_seed"] = 21
        plan.update({"selection_seeds": selection_seeds, "confirmation_seeds": confirmation_seeds,
                     "confirmation_models": [], "run_schedule": schedule,
                     "implementation_hashes": current,
                     "selection_rule": "Lowest mean validation MAE over candidate runs at seeds 7 and 21 only; user cancelled all later-seed training",
                     "revision": {"at_utc": datetime.now(timezone.utc).isoformat(),
                                  "reason": "User superseded sparse confirmation schedule with two seeds only, preserving completed work and finishing seed 21 variants",
                                  "original_plan": str(original.resolve()),
                                  "preserved_completed_runs": {seed: {name: record["checkpoint_sha256"] for name, record in models.items()} for seed, models in records.items()},
                                  "cancelled_seeds": [42, 87, 123],
                                  "deployment_seed": "21, chosen by schedule rather than test performance"}})
        write_json(directory / "plan.json", plan)
        handoff["status"] = "plan_revised_ready_to_resume"
        write_json(directory / "handoff.json", handoff)
        print("Revised study: 12 runs, seeds 7/21 only. All completed checkpoints preserved; finish seed 21's remaining variants.")


if __name__ == "__main__":
    main()
