"""Run RNO Darcy inference, metric evaluation, result saving, and visualization."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import FieldNormalizer, build_loaders, synthetic_loaders
from model import RadonNeuralOperator2D
from utils import gradient_loss, load_config, move_batch, prepare_output_dirs, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--config", default=None, help="Optional config override; checkpoint config is default")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic smoke data")
    parser.add_argument("--no-visualization", action="store_true", help="Skip PNG generation")
    return parser.parse_args()


def plot_training_curve(history_path: Path, output_dir: Path) -> Optional[Path]:
    if not history_path.is_file():
        return None
    with history_path.open("r", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        return None
    epochs = np.asarray([int(row["epoch"]) for row in records])
    loss = np.asarray([float(row["train_loss"]) for row in records])
    relative = np.asarray([float(row["train_relative_l2"]) for row in records])
    gradient = np.asarray([float(row["train_gradient_relative_l2"]) for row in records])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(epochs, loss, label="total loss")
    axis.semilogy(epochs, relative, label="relative L2")
    axis.semilogy(epochs, gradient, label="gradient relative L2")
    axis.set(xlabel="Epoch", ylabel="Metric", title="RNO Darcy Flow training")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output = output_dir / "training_curve.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_prediction(
    coefficient: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_dir: Path,
) -> Path:
    fields = [coefficient[0, 0], target[0, 0], prediction[0, 0], np.abs(prediction[0, 0] - target[0, 0])]
    titles = ["Normalized coefficient", "Target", "Prediction", "Absolute error"]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, field, title in zip(axes, fields, titles):
        image = axis.imshow(field, origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output = output_dir / "darcy_prediction.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    # Keep non-model checkpoint state (notably normalization statistics) on CPU.
    # load_state_dict copies model tensors to the device selected below.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: Dict[str, Any] = checkpoint["config"]
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config = load_config(config_path)
    if args.output_dir:
        config["output"]["root"] = args.output_dir
    paths = prepare_output_dirs(config, PROJECT_ROOT)

    input_norm = FieldNormalizer.from_state_dict(checkpoint["input_normalizer"])
    output_norm = FieldNormalizer.from_state_dict(checkpoint["output_normalizer"])
    if args.smoke:
        _, test_loader, _, _ = synthetic_loaders(
            resolution=int(config["data"]["resolution"]),
            batch_size=int(config["evaluation"]["batch_size"]),
        )
    else:
        _, test_loader, _, _ = build_loaders(
            config,
            args.data_root,
            input_normalizer=input_norm,
            output_normalizer=output_norm,
        )

    model = RadonNeuralOperator2D.from_config(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    ratios, gradient_ratios = [], []
    predictions, targets, coefficients = [], [], []
    saved_samples = 0
    max_saved = int(config["evaluation"].get("max_saved_samples", 0))
    save_predictions = bool(config["evaluation"].get("save_predictions", True))
    started = time.time()
    with torch.no_grad():
        for coefficient, target in test_loader:
            coefficient, target = move_batch((coefficient, target), device)
            prediction = output_norm.decode(model(coefficient))
            flat_error = (prediction - target).reshape(prediction.shape[0], -1)
            flat_target = target.reshape(target.shape[0], -1)
            batch_ratios = torch.linalg.vector_norm(flat_error, dim=1) / torch.linalg.vector_norm(
                flat_target, dim=1
            ).clamp_min(1.0e-12)
            ratios.extend(batch_ratios.cpu().tolist())
            gradient_ratios.append(float(gradient_loss(prediction, target)))
            if save_predictions and saved_samples < max_saved:
                take = min(prediction.shape[0], max_saved - saved_samples)
                predictions.append(prediction[:take].cpu().numpy())
                targets.append(target[:take].cpu().numpy())
                coefficients.append(coefficient[:take].cpu().numpy())
                saved_samples += take

    test_relative_l2 = float(np.mean(ratios))
    paper_target = float(config["experiment"]["paper_target_relative_l2"])
    metrics = {
        "metric": "mean_per_sample_relative_l2",
        "test_relative_l2": test_relative_l2,
        "test_relative_l2_std": float(np.std(ratios)),
        "test_samples": len(ratios),
        "test_gradient_relative_l2_batch_mean": float(np.mean(gradient_ratios)),
        "paper_target_relative_l2": paper_target,
        "absolute_gap_to_paper": abs(test_relative_l2 - paper_target),
        "ratio_to_paper": test_relative_l2 / paper_target,
        "matches_paper_within_10_percent": abs(test_relative_l2 - paper_target) / paper_target <= 0.10,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "elapsed_seconds": time.time() - started,
    }
    write_json(metrics, paths["results"] / "evaluation_metrics.json")

    prediction_array = target_array = coefficient_array = None
    if predictions:
        prediction_array = np.concatenate(predictions)
        target_array = np.concatenate(targets)
        coefficient_array = np.concatenate(coefficients)
        np.savez_compressed(
            paths["results"] / "predictions.npz",
            prediction=prediction_array,
            target=target_array,
            normalized_coefficient=coefficient_array,
            sample_relative_l2=np.asarray(ratios[:saved_samples]),
        )

    visualization = {
        "training_curve": None,
        "prediction_figure": None,
        "skipped": bool(args.no_visualization),
    }
    if not args.no_visualization:
        curve = plot_training_curve(paths["logs"] / "train.csv", paths["results"])
        visualization["training_curve"] = str(curve) if curve else None
        if prediction_array is not None and target_array is not None and coefficient_array is not None:
            figure = plot_prediction(coefficient_array, target_array, prediction_array, paths["results"])
            visualization["prediction_figure"] = str(figure)
    visualization.update(
        test_relative_l2=test_relative_l2,
        paper_target_relative_l2=paper_target,
        matches_paper_within_10_percent=metrics["matches_paper_within_10_percent"],
    )
    write_json(visualization, paths["results"] / "visualization_summary.json")
    print({"metrics": metrics, "visualization": visualization})


if __name__ == "__main__":
    main()
