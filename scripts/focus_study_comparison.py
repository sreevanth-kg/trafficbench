#!/usr/bin/env python3
"""Preserve completed work and focus the study on Graph WaveNet versus ours."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from trafficbench.discovery import sha256
from trafficbench.study import implementation_hashes, read_records, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.output)
    with (directory / ".study.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((directory / "plan.json").read_text())
        if (directory / "selection.json").exists():
            raise ValueError("Architecture selection has already been frozen")
        if plan["implementation_hashes"] != implementation_hashes():
            raise ValueError("Implementation differs from the revised plan")
        records = read_records(directory, plan["config"]["seeds"])
        for seed, models in records.items():
            for name, record in models.items():
                if sha256(directory / "seeds" / f"seed-{seed}" / f"{name}.pt") != record["checkpoint_sha256"]:
                    raise ValueError("A completed checkpoint changed")
        backup = directory / "plan.two-seed-ablations.json"
        if backup.exists():
            raise ValueError("Focused plan was already applied")
        write_json(backup, plan)
        previous = plan["revision"]
        plan["config"]["baselines"] = ["graphwavenet"]
        plan["config"]["candidates"] = ["regimeliquid"]
        plan["run_schedule"] = [{"seed": seed, "model": name} for seed in (7, 21)
                                for name in ("graphwavenet", "regimeliquid")]
        plan["selection_rule"] = "User-fixed regimeliquid architecture; per-seed checkpoints remain selected by validation only; compare seeds 7 and 21"
        plan["revision"] = {"at_utc": datetime.now(timezone.utc).isoformat(),
                            "reason": "User requested only Graph WaveNet and ours at two seeds",
                            "previous_revision": previous, "previous_plan": str(backup.resolve()),
                            "preserved_completed_runs": {seed: list(models) for seed, models in records.items()}}
        write_json(directory / "plan.json", plan)
        print("Focused schedule: Graph WaveNet and regimeliquid at seeds 7/21; all four fits already complete. Additional fits cancelled, completed artifacts preserved.")


if __name__ == "__main__":
    main()
