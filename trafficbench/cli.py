"""Command-line entry points for forecasting and simulated routing."""

import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    forecast = commands.add_parser("forecast", help="Train/evaluate METR-LA baselines.")
    forecast.add_argument("--data", default="data/metr-la")
    forecast.add_argument("--output", default=f"results/metr-la/{datetime.now():%Y%m%d-%H%M%S}")
    forecast.add_argument("--models", nargs="+", choices=["persistence", "lstm", "graphwavenet"], default=["persistence", "lstm", "graphwavenet"])
    forecast.add_argument("--epochs", type=int, default=50)
    forecast.add_argument("--patience", type=int, default=10)
    forecast.add_argument("--batch-size", type=int, default=64)
    forecast.add_argument("--learning-rate", type=float, default=0.001)
    forecast.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    forecast.add_argument("--threads", type=int, default=4)
    forecast.add_argument("--seed", type=int, default=42)
    forecast.add_argument("--congestion-mph", type=float, default=40.0)
    for split in ("train", "val", "test"):
        forecast.add_argument(f"--max-{split}", type=int, default=0, help="0 uses the full split; positive caps label a smoke run.")
    routing = commands.add_parser("route", help="Evaluate policies on an abstract METR-LA sensor graph.")
    routing.add_argument("--data", default="data/metr-la")
    routing.add_argument("--forecasts", required=True)
    routing.add_argument("--pairs", type=int, default=20)
    routing.add_argument("--departures", type=int, default=24)
    routing.add_argument("--departure-minutes", type=int, choices=[15, 30, 60], default=15)
    routing.add_argument("--speed-floor-mph", type=float, default=5.0)
    routing.add_argument("--seed", type=int, default=42)
    discovery = commands.add_parser("discover", help="Select the proposed architecture/ablations on validation only.")
    discovery.add_argument("--data", default="data/metr-la")
    discovery.add_argument("--output", default="results/metr-la/proposed-10ep")
    discovery.add_argument("--candidates", nargs="+", choices=["regimeliquid", "liquidgraph", "regimegru", "regimeliquid_noprior", "rglflow_msr"],
                           default=["regimeliquid", "liquidgraph", "regimegru", "regimeliquid_noprior"])
    discovery.add_argument("--epochs", type=int, default=10)
    discovery.add_argument("--patience", type=int, default=5)
    discovery.add_argument("--batch-size", type=int, default=64)
    discovery.add_argument("--learning-rate", type=float, default=0.001)
    discovery.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    discovery.add_argument("--threads", type=int, default=4)
    discovery.add_argument("--seed", type=int, default=42)
    discovery.add_argument("--max-train", type=int, default=0)
    discovery.add_argument("--max-val", type=int, default=0)
    selected = commands.add_parser("evaluate-selected", help="Evaluate the frozen validation choice and export an inference bundle.")
    selected.add_argument("--data", default="data/metr-la")
    selected.add_argument("--experiment", default="results/metr-la/proposed-10ep")
    selected.add_argument("--baseline", default="results/metr-la/baseline-10ep")
    selected.add_argument("--batch-size", type=int, default=64)
    selected.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    selected.add_argument("--threads", type=int, default=4)
    prediction = commands.add_parser("predict", help="Forecast a history-only JSON request with an exported bundle.")
    prediction.add_argument("--bundle", default="results/metr-la/proposed-10ep/bundle")
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--output", default="results/metr-la/prediction.json")
    prediction.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    prediction.add_argument("--threads", type=int, default=4)
    service = commands.add_parser("serve-forecast", help="Serve the exported bundle as a local HTTP demo.")
    service.add_argument("--bundle", default="results/metr-la/proposed-10ep/bundle")
    service.add_argument("--host", default="127.0.0.1")
    service.add_argument("--port", type=int, default=8000)
    service.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    service.add_argument("--threads", type=int, default=4)
    study = commands.add_parser("study", help="Run/resume a matched multi-seed study with validation-only architecture selection.")
    study.add_argument("--data", default="data/metr-la")
    study.add_argument("--output", default="results/metr-la/study-50ep-5seed")
    study.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 87, 123])
    study.add_argument("--candidates", nargs="+", choices=["regimeliquid", "liquidgraph", "regimegru", "regimeliquid_noprior", "rglflow_msr"],
                       default=["regimeliquid", "liquidgraph", "regimegru", "regimeliquid_noprior"])
    study.add_argument("--baselines", nargs="+", choices=["lstm", "graphwavenet"], default=["lstm", "graphwavenet"])
    study.add_argument("--epochs", type=int, default=50)
    study.add_argument("--patience", type=int, default=10)
    study.add_argument("--batch-size", type=int, default=64)
    study.add_argument("--learning-rate", type=float, default=0.001)
    study.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    study.add_argument("--threads", type=int, default=4)
    study.add_argument("--stage", choices=["all", "train", "evaluate", "status"], default="all")
    study.add_argument("--resume", action="store_true")
    study.add_argument("--deployment-seed", type=int)
    study.add_argument("--pairs", type=int, default=20)
    study.add_argument("--departures", type=int, default=24)
    for split in ("train", "val", "test"):
        study.add_argument(f"--max-{split}", type=int, default=0, help="Smoke subset limit; 0 uses the full split.")
    args = parser.parse_args()
    try:
        if args.command == "forecast":
            from .experiment import run
        elif args.command == "route":
            from .routing import run
        elif args.command == "discover":
            from .discovery import discover as run
        elif args.command == "evaluate-selected":
            from .discovery import evaluate_selected as run
        elif args.command == "predict":
            from .inference import run
        elif args.command == "study":
            from .study import run
        else:
            from .service import run
        run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
