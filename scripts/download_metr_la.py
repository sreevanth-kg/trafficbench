#!/usr/bin/env python3
"""Download pinned METR-LA or PEMS-BAY observations and graph artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


METR_H5_URL = (
    "https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/"
    "ec9772c0e8dd48b3470a6381dff1888257a9ffe4/metr-la.h5"
)
METR_H5_SHA256 = "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"
PEMS_H5_URL = (
    "https://huggingface.co/datasets/jimmygao3218/PEMSBAY/resolve/"
    "c858397fb381d82c3d55ad0a81beb51a9bb9bdaa/PEMS-BAY.h5"
)
PEMS_H5_SHA256 = "65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f"
DCRNN_COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
GRAPH_BASE = f"https://raw.githubusercontent.com/liyaguang/DCRNN/{DCRNN_COMMIT}/data/sensor_graph/"

DATASETS = {
    "metr-la": {
        "name": "METR-LA", "observation_file": "metr-la.h5", "h5_url": METR_H5_URL,
        "hdf_key": "df",
        "h5_sha256": METR_H5_SHA256, "adjacency_file": "adj_mx.pkl",
        "distance_file": "distances_la_2012.csv", "locations_file": "graph_sensor_locations.csv",
    },
    "pems-bay": {
        "name": "PEMS-BAY", "observation_file": "pems-bay.h5", "h5_url": PEMS_H5_URL,
        "hdf_key": "speed",
        "h5_sha256": PEMS_H5_SHA256, "adjacency_file": "adj_mx_bay.pkl",
        "distance_file": "distances_bay_2017.csv", "locations_file": "graph_sensor_locations_bay.csv",
    },
}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(directory, dataset="metr-la"):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}")
    settings = DATASETS[dataset]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    files = [(settings["observation_file"], settings["h5_url"], settings["h5_sha256"])] + [
        (name, GRAPH_BASE + name, None)
        for name in (settings["adjacency_file"], settings["distance_file"], settings["locations_file"])
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
        "dataset": settings["name"],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "observations_source": "Pinned Hugging Face mirror of the DCRNN raw HDF5 dataset; original source is the DCRNN Google Drive release",
        "graph_source_commit": DCRNN_COMMIT,
        "observation_file": settings["observation_file"],
        "hdf_key": settings["hdf_key"],
        "adjacency_file": settings["adjacency_file"],
        "distance_file": settings["distance_file"],
        "locations_file": settings["locations_file"],
        "artifacts": artifacts,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {settings['name']} to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="metr-la")
    parser.add_argument("--directory")
    args = parser.parse_args()
    download(args.directory or f"data/{args.dataset}", args.dataset)
