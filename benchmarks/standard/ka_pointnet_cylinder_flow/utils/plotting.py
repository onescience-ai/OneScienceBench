from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_field_triptych(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    field_name: str,
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    error = np.abs(prediction - target)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = [
        (prediction, f"Predicted {field_name}"),
        (target, f"Ground Truth {field_name}"),
        (error, f"Absolute Error {field_name}"),
    ]
    for axis, (values, title) in zip(axes, panels):
        scatter = axis.scatter(x_coords, y_coords, c=values, s=5.0, cmap="jet")
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(scatter, ax=axis)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
