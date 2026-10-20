from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(results_dir: str | Path = "results") -> dict:
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = {
        "relative_l2": metrics.get("relative_l2"),
        "mse": metrics.get("mse"),
        "passed_smoke": metrics.get("relative_l2") is not None,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    summarize(args.results_dir)


if __name__ == "__main__":
    main()
