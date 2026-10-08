"""Raw-time splits, causal input imputation and memory-efficient forecasting windows."""

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


STEP_NS = 300_000_000_000


def split_origins(timestamps, history=12, horizon=12):
    """Keep all labels inside one split and reject windows spanning timestamp gaps."""
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    size = len(timestamps)
    if size < history + horizon + 3:
        raise ValueError("Not enough observations for chronological forecasting splits.")
    train_end, val_end = int(size * 0.7), int(size * 0.8)
    origins = np.arange(history - 1, size - horizon)
    gaps = np.r_[0, np.cumsum(np.diff(timestamps.astype(np.int64)) != STEP_NS)]
    origins = origins[gaps[origins + horizon] == gaps[origins - history + 1]]
    splits = {
        "train": origins[origins + horizon < train_end],
        "val": origins[(origins + 1 >= train_end) & (origins + horizon < val_end)],
        "test": origins[origins + 1 >= val_end],
    }
    if any(len(values) == 0 for values in splits.values()):
        raise ValueError("A forecasting split has no valid windows.")
    return splits, train_end, val_end


def causal_fill(raw, train_end):
    """Forward-fill inputs only; fallback and normalization use training data only."""
    valid = np.isfinite(raw) & (raw > 0)
    training = np.where(valid[:train_end], raw[:train_end], np.nan)
    mean = float(np.nanmean(training))
    std = max(float(np.nanstd(training)), 1e-6)
    if not np.isfinite(mean):
        raise ValueError("Training observations contain no valid speed values.")
    means = np.nanmean(training, axis=0)
    means = np.where(np.isfinite(means), means, mean).astype(np.float32)
    frame = pd.DataFrame(np.where(valid, raw, np.nan))
    filled = frame.ffill().fillna(pd.Series(means)).to_numpy(dtype=np.float32)
    return filled, valid, mean, std, means


@dataclass
class TrafficData:
    raw: np.ndarray
    filled: np.ndarray
    valid: np.ndarray
    features: np.ndarray
    timestamps: np.ndarray
    sensor_ids: list[str]
    adjacency: np.ndarray
    origins: dict[str, np.ndarray]
    train_end: int
    val_end: int
    mean: float
    std: float
    static_speeds: np.ndarray
    history: int = 12
    horizon: int = 12
    dataset: str = "METR-LA"
    distance_file: str = "distances_la_2012.csv"


def load_data(directory, history=12, horizon=12):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    observation_file = manifest.get("observation_file", "metr-la.h5")
    adjacency_file = manifest.get("adjacency_file", "adj_mx.pkl")
    # The original dataset uses legacy pandas HDF5 attributes that modern
    # pandas/PyTables can decode inconsistently. Read its fixed-format numeric
    # block directly, preserving the explicit column and datetime axes.
    with h5py.File(directory / observation_file, "r") as store:
        hdf_key = manifest.get("hdf_key", "df")
        if hdf_key not in store and "hdf_key" not in manifest:
            groups = [key for key, value in store.items() if isinstance(value, h5py.Group)]
            if len(groups) != 1:
                raise ValueError(f"Cannot identify the observation group in {observation_file}.")
            hdf_key = groups[0]
        if hdf_key not in store:
            raise ValueError(f"Expected HDF5 key {hdf_key!r} for {manifest['dataset']}")
        group = store[hdf_key]
        decode = lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
        columns = [decode(value) for value in group["axis0"][:]]
        block_columns = [decode(value) for value in group["block0_items"][:]]
        if set(columns) != set(block_columns):
            raise ValueError(f"Unexpected multi-block {manifest['dataset']} HDF5 structure.")
        frame = pd.DataFrame(group["block0_values"][:],
                             index=pd.to_datetime(group["axis1"][:], unit="ns"),
                             columns=block_columns).reindex(columns=columns)
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{manifest['dataset']} needs a unique, increasing datetime index.")
    sensor_ids = [str(value) for value in frame.columns]
    if len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("Duplicate sensor columns.")
    # This is the pinned, original DCRNN graph artifact downloaded by our bootstrap.
    with (directory / adjacency_file).open("rb") as stream:
        graph_ids, _mapping, adjacency = pickle.load(stream, encoding="latin1")
    graph_lookup = {decode(sensor): position for position, sensor in enumerate(graph_ids)}
    if set(sensor_ids) != set(graph_lookup):
        raise ValueError("Observation sensor IDs do not match the adjacency artifact.")
    ordering = [graph_lookup[sensor] for sensor in sensor_ids]
    adjacency = np.asarray(adjacency, dtype=np.float32)[np.ix_(ordering, ordering)]
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("Adjacency must contain finite nonnegative values.")
    raw = frame.to_numpy(dtype=np.float32)
    timestamps = frame.index.to_numpy(dtype="datetime64[ns]")
    origins, train_end, val_end = split_origins(timestamps, history, horizon)
    filled, valid, mean, std, means = causal_fill(raw, train_end)
    fraction_of_day = (frame.index.hour * 60 + frame.index.minute).to_numpy(dtype=np.float32) / 1440
    features = np.stack(
        [(filled - mean) / std, np.broadcast_to(fraction_of_day[:, None], raw.shape)], axis=-1,
    ).astype(np.float32)
    return TrafficData(raw, filled, valid, features, timestamps, sensor_ids, adjacency,
                       origins, train_end, val_end, mean, std, means, history, horizon, manifest["dataset"],
                       manifest.get("distance_file", "distances_la_2012.csv"))
def limit_origins(origins, limit):
    """Spread a smoke-run subset across its split, rather than using only one day."""
    if limit > 0 and len(origins) > limit:
        return origins[np.linspace(0, len(origins) - 1, limit, dtype=int)]
    return origins.copy()


class Windows(Dataset):
    def __init__(self, data, origins):
        self.data = data
        self.origins = origins

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        t = self.origins[index]
        x = self.data.features[t - self.data.history + 1:t + 1]
        y = self.data.raw[t + 1:t + self.data.horizon + 1]
        mask = self.data.valid[t + 1:t + self.data.horizon + 1]
        return torch.from_numpy(x), torch.from_numpy(np.where(mask, y, 0)), torch.from_numpy(mask)
