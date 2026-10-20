from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ensure_dir, load_config
from utils.plotting import save_field_triptych


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize KA-PointNet predictions for one sample.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--split", default=None, choices=["train", "validation", "test"])
    parser.add_argument("--sample-index", type=int, default=None)
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_name = args.split or config["inference"]["split"]
    sample_index = int(args.sample_index if args.sample_index is not None else config["visualization"]["sample_index"])

    results_dir = project_path(config["experiment"]["results_dir"])
    prediction_path = results_dir / f"{split_name}_predictions.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}. Run scripts/inference.py first.")

    payload = np.load(prediction_path)
    predictions = payload["predictions"]
    targets = payload["targets"]
    coordinates = payload["coordinates"]

    x_coords = coordinates[sample_index, :, 0]
    y_coords = coordinates[sample_index, :, 1]

    figure_dir = ensure_dir(results_dir / "figures" / split_name)
    field_names = ["u_velocity", "v_velocity", "pressure"]
    for channel, field_name in enumerate(field_names):
        save_field_triptych(
            x_coords=x_coords,
            y_coords=y_coords,
            prediction=predictions[sample_index, channel, :],
            target=targets[sample_index, channel, :],
            field_name=field_name,
            output_path=figure_dir / f"{field_name}_sample_{sample_index}.png",
            dpi=int(config["visualization"]["dpi"]),
        )

    print(f"Saved result figures under {figure_dir}")


if __name__ == "__main__":
    main()
