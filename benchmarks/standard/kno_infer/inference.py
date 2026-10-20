"""Standalone KNO inference for the Navier--Stokes benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.ticker import FormatStrFormatter

from dataset import NavierStokesDatapipe
from model import KNO2DNavierStokes
from utils import load_config, project_path, relative_l2, resolve_device, rollout


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.device is not None:
        config["common"]["device"] = args.device
    if args.data is not None:
        data_path = Path(args.data).expanduser().resolve()
        config["datapipe"]["source"].update({"data_dir": str(data_path.parent), "file_name": data_path.name})
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        config["training"].update({"weight_dir": str(checkpoint.parent), "checkpoint_name": checkpoint.name})
    if args.output is not None:
        config["inference"]["result_dir"] = args.output
    if args.num_samples is not None:
        config["inference"]["num_samples"] = args.num_samples
    if args.max_batches is not None:
        config["inference"]["max_batches"] = args.max_batches


def resolve_data_path(config: dict[str, Any]) -> Path:
    source = config["datapipe"]["source"]
    source["data_dir"] = str(project_path(source["data_dir"], PROJECT_ROOT).resolve())
    return Path(source["data_dir"]) / source["file_name"]


def validate_checkpoint_config(config: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    saved = checkpoint.get("datapipe_config", checkpoint.get("data_config"))
    if saved is None:
        raise KeyError("Checkpoint contains no datapipe configuration")
    current_data, saved_data = config["datapipe"]["data"], saved["data"]
    keys = ("t_in", "t_out", "out_dim", "downsamplex", "downsampley", "normalize")
    mismatches = [key for key in keys if current_data[key] != saved_data[key]]
    if mismatches:
        details = ", ".join(f"{key}: config={current_data[key]!r}, checkpoint={saved_data[key]!r}" for key in mismatches)
        raise ValueError(f"Checkpoint/data configuration mismatch: {details}")


def build_model(model_config: dict[str, Any], spatial_shape: tuple[int, int], datapipe_config: dict[str, Any]) -> KNO2DNavierStokes:
    data = datapipe_config["data"]
    return KNO2DNavierStokes(
        input_channels=int(model_config.get("input_channels", data["t_in"] * data["out_dim"])),
        output_channels=int(model_config.get("output_channels", data["out_dim"])),
        spatial_shape=spatial_shape,
        op_size=int(model_config["op_size"]),
        modes_x=int(model_config["modes_x"]),
        modes_y=int(model_config["modes_y"]),
        decompose=int(model_config["decompose"]),
        linear_type=bool(model_config["linear_type"]),
        normalization=bool(model_config["normalization"]),
    )


def save_visualization(prediction: torch.Tensor, target: torch.Tensor, spatial_shape: tuple[int, int], output_path: Path) -> None:
    prediction, target = prediction.detach().cpu(), target.detach().cpu()
    height, width = spatial_shape
    frame_count = prediction.shape[-1]
    frames = sorted({0, frame_count // 2, frame_count - 1})
    sample_prediction = prediction[0].reshape(height, width, frame_count)
    sample_target = target[0].reshape(height, width, frame_count)
    rows = (("Target", sample_target), ("Prediction", sample_prediction), ("Abs Error", (sample_prediction - sample_target).abs()))
    figure, axes = plt.subplots(3, len(frames), figsize=(4 * len(frames), 9), squeeze=False)
    for row_index, (title, values) in enumerate(rows):
        for column_index, frame in enumerate(frames):
            axis = axes[row_index][column_index]
            image = axis.imshow(values[..., frame], origin="lower", cmap="viridis")
            axis.set_title(f"{title} t+{frame + 1}")
            axis.set_xticks([])
            axis.set_yticks([])
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.formatter = FormatStrFormatter("%.3g")
            colorbar.update_ticks()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", help="Direct path to an original KNO checkpoint")
    parser.add_argument("--data", help="Direct path to NavierStokes_V1e-5_N1200_T20.mat")
    parser.add_argument("--output", help="Inference result directory")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--num-samples", type=int, help="Number of predictions saved")
    parser.add_argument("--max-batches", type=int, help="Limit evaluated test batches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    apply_overrides(config, args)
    data_path = resolve_data_path(config)
    training, inference = config["training"], config["inference"]
    checkpoint_path = project_path(training["weight_dir"], PROJECT_ROOT).resolve() / str(training["checkpoint_name"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_config(config, checkpoint)
    device = resolve_device(str(config["common"]["device"]))

    datapipe = NavierStokesDatapipe(config["datapipe"], normalizer_state=checkpoint.get("normalizer"))
    spatial_shape = tuple(int(value) for value in checkpoint["spatial_shape"])
    if datapipe.spatial_shape != spatial_shape:
        raise ValueError(f"Checkpoint grid {spatial_shape} does not match data grid {datapipe.spatial_shape}")
    model = build_model(checkpoint["model_config"], spatial_shape, config["datapipe"]).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incompatible checkpoint: {incompatible}")
    model.eval()

    max_batches = inference.get("max_batches")
    max_batches = None if max_batches is None else int(max_batches)
    if max_batches is not None and max_batches < 1:
        raise ValueError("inference.max_batches must be null or positive")
    num_samples = int(inference["num_samples"])
    if num_samples < 1:
        raise ValueError("inference.num_samples must be positive")

    squared_error = relative_error = 0.0
    element_count = sample_count = 0
    saved_prediction: list[torch.Tensor] = []
    saved_target: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(datapipe.test_dataloader()):
            if max_batches is not None and batch_index >= max_batches:
                break
            pos, state, target = (batch[key].to(device) for key in ("pos", "x", "y"))
            prediction = rollout(model, pos, state, int(config["datapipe"]["data"]["t_out"]), int(config["datapipe"]["data"]["out_dim"]))
            prediction = datapipe.decode_solution(prediction)
            target = datapipe.decode_solution(target)
            squared_error += F.mse_loss(prediction, target, reduction="sum").item()
            element_count += target.numel()
            relative_error += relative_l2(prediction, target).sum().item()
            sample_count += target.shape[0]
            remaining = num_samples - sum(item.shape[0] for item in saved_prediction)
            if remaining > 0:
                saved_prediction.append(prediction[:remaining].cpu())
                saved_target.append(target[:remaining].cpu())
    if sample_count == 0:
        raise RuntimeError("No test samples were evaluated")

    prediction_sample, target_sample = torch.cat(saved_prediction), torch.cat(saved_target)
    metrics = {
        "checkpoint": str(checkpoint_path), "data": str(data_path), "device": str(device),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_relative_l2": float(checkpoint.get("best_relative_l2", float("nan"))),
        "sample_count": sample_count, "mse": squared_error / element_count,
        "relative_l2": relative_error / sample_count, "spatial_shape": list(spatial_shape),
        "t_in": int(config["datapipe"]["data"]["t_in"]), "t_out": int(config["datapipe"]["data"]["t_out"]),
    }
    result_dir = project_path(inference["result_dir"], PROJECT_ROOT).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"prediction": prediction_sample, "target": target_sample, "spatial_shape": spatial_shape, "mse": metrics["mse"], "relative_l2": metrics["relative_l2"]}, result_dir / "prediction_sample.pt")
    save_visualization(prediction_sample[:1], target_sample[:1], spatial_shape, result_dir / "prediction_sample.png")
    with (result_dir / "inference_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_path}")
    print(f"Device: {device}")
    print(f"Samples: {sample_count}")
    print(f"MSE: {metrics['mse']:.6e}")
    print(f"Relative L2: {metrics['relative_l2']:.6e}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
