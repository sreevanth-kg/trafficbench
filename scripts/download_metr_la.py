#!/usr/bin/env python3
"""Download pinned METR-LA observations and DCRNN sensor graph artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


H5_URL = (
    "https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/"
    "ec9772c0e8dd48b3470a6381dff1888257a9ffe4/metr-la.h5"
)
H5_SHA256 = "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"
DCRNN_COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
GRAPH_BASE = f"https://raw.githubusercontent.com/liyaguang/DCRNN/{DCRNN_COMMIT}/data/sensor_graph/"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    files = [("metr-la.h5", H5_URL, H5_SHA256)] + [
        (name, GRAPH_BASE + name, None)
        for name in ("adj_mx.pkl", "distances_la_2012.csv", "graph_sensor_locations.csv")
    ]
    for name, url, expected in files:
        target = directory / name
        if not target.exists() or (expected and sha256(target) != expected):
            print(f"Downloading {name}", flush=True)
            partial = target.with_suffix(target.suffix + ".part")
            try:
                request = Request(url, headers={"User-Agent": "trafficbench/0.1"})
                with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if expected and sha256(partial) != expected:
                    raise ValueError(f"Checksum mismatch: {name}")
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        artifacts.append({"file": name, "url": url, "sha256": sha256(target), "bytes": target.stat().st_size})
    manifest = {
        "dataset": "METR-LA",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "observations_source": "Pinned Hugging Face mirror of the DCRNN raw HDF5 dataset",
        "graph_source_commit": DCRNN_COMMIT,
        "artifacts": artifacts,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded METR-LA to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/metr-la")
    download(parser.parse_args().directory)
