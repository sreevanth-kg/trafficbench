#!/usr/bin/env python3
"""Real-bundle HTTP integration check on an ephemeral local port."""

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from trafficbench.inference import ForecastEngine
from trafficbench.service import make_handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="results/metr-la/proposed-10ep/bundle")
    args = parser.parse_args()
    engine = ForecastEngine(args.bundle, "cpu")
    request = json.loads((Path(args.bundle) / "example_request.json").read_text())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/health", timeout=10) as response:
            assert json.load(response)["status"] == "ready"
        payload = Request(url + "/forecast", data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
        with urlopen(payload, timeout=30) as response:
            result = json.load(response)
        median, lower, upper = [np.asarray(result[field]) for field in ("median_speeds_mph", "lower_speeds_mph", "upper_speeds_mph")]
        assert median.shape == (12, len(engine.metadata["sensor_ids"]))
        assert np.isfinite(median).all() and (lower >= 0).all()
        assert (lower <= median).all() and (median <= upper).all()
        np.testing.assert_allclose(np.asarray(result["regime_weights"]).sum(axis=-1), 1, atol=2e-7)
        direct = engine.predict(request)
        np.testing.assert_allclose(median, direct["median_speeds_mph"], rtol=1e-6, atol=1e-6)
        # The exported example is the middle test origin. Check the API against
        # independently saved batched evaluation, including CPU/GPU tolerance.
        evaluated = Path(args.bundle).parent / "evaluation" / engine.metadata.get("evaluation_forecast_filename", f"{engine.metadata['architecture']}_forecasts.npz")
        with np.load(evaluated, allow_pickle=False) as saved:
            assert saved["sensor_ids"].tolist() == result["sensor_ids"]
            forecasts = saved["prediction"]
            np.testing.assert_allclose(median, forecasts[len(forecasts) // 2], rtol=1e-5, atol=1e-4)
        bad = Request(url + "/forecast", data=b'{"sensor_ids": ["wrong-network"]}', headers={"Content-Type": "application/json"})
        try:
            urlopen(bad, timeout=10)
            raise AssertionError("Invalid sensor network was accepted")
        except HTTPError as error:
            assert error.code == 400
        print(f"HTTP integration passed: {median.shape}, saved-evaluation agreement, ordered intervals, normalized gates, invalid-network rejection.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
