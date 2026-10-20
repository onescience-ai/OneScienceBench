from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(results_dir: Path) -> dict:
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.is_file():
        summary = {"available": False, "reason": "metrics.json missing"}
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = {"available": True, "metrics": metrics}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    summarize(Path(args.results_dir))


if __name__ == "__main__":
    main()
