"""Train the SFNO and its H^-1 Spectral-Refiner for paper Table 2.

Normal mode follows the paper repository notebooks for the forced-turbulence
FNO data.  ``--smoke`` keeps the same execution path with a tiny deterministic
trajectory so the complete artifact contract can be checked cheaply.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from model import OutConvFT, SFNO, SobolevLoss


ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("config.yaml must contain a mapping")
    return config


def resolve_path(raw_path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) if base_dir is not None else ROOT).joinpath(path).resolve()


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_trajectory_file(
    path: str | Path,
    field: str,
    input_steps: int,
    output_steps: int,
    start_step: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse an official trajectory dictionary into time-last model tensors."""
    payload = _torch_load(Path(path))
    if not isinstance(payload, dict) or field not in payload:
        raise KeyError(f"trajectory file must contain a '{field}' tensor")
    values = payload[field]
    if not isinstance(values, torch.Tensor) or values.ndim != 4:
        raise ValueError(f"{field} must have shape [N,T,X,Y] or [N,X,Y,T]")

    required_steps = start_step + input_steps + output_steps
    if values.shape[1] >= required_steps and values.shape[2] == values.shape[3]:
        values = values.permute(0, 2, 3, 1)
    elif values.shape[-1] < required_steps or values.shape[1] != values.shape[2]:
        raise ValueError(f"cannot identify time dimension in shape {tuple(values.shape)}")

    values = values.to(dtype=dtype).contiguous()
    inputs = values[..., start_step : start_step + input_steps]
    targets = values[
        ..., start_step + input_steps : start_step + input_steps + output_steps
    ]
    if inputs.shape[-1] != input_steps or targets.shape[-1] != output_steps:
        raise ValueError("trajectory does not contain the requested temporal window")
    return inputs, targets


def build_model(config: dict[str, Any]) -> SFNO:
    model_cfg = config["model"]
    data_cfg = config["data"]
    return SFNO(
        modes_x=int(model_cfg["modes_x"]),
        modes_y=int(model_cfg["modes_y"]),
        modes_t=int(model_cfg["modes_t"]),
        width=int(model_cfg["width"]),
        beta=float(model_cfg["beta"]),
        num_spectral_layers=int(model_cfg["num_spectral_layers"]),
        latent_steps=int(model_cfg.get("latent_steps", data_cfg["input_steps"])),
        activation=str(model_cfg.get("activation", "ReLU")),
        spatial_padding=int(model_cfg.get("spatial_padding", 0)),
        output_steps=int(data_cfg["output_steps"]),
    )


def forcing_field(n_grid: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    grid = torch.arange(n_grid, device=device, dtype=dtype) / n_grid
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    return 0.1 * (torch.sin(2 * math.pi * (x + y)) + torch.cos(2 * math.pi * (x + y)))


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.linalg.vector_norm((prediction - target).reshape(prediction.shape[0], -1), dim=1)
    denom = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-12)
    return (diff / denom).mean()


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["data"].update(
        {
            "train_samples": 4,
            "validation_samples": 2,
            "test_samples": 1,
            "train_resolution": 8,
            "evaluation_resolution": 8,
            "input_steps": 4,
            "output_steps": 4,
            "test_start_step": 0,
        }
    )
    result["model"].update(
        {
            "modes_x": 2,
            "modes_y": 2,
            "modes_t": 2,
            "width": 4,
            "num_spectral_layers": 2,
            "latent_steps": 4,
        }
    )
    result["training"].update({"epochs": 1, "batch_size": 2})
    result["fine_tuning"].update(
        {"iterations": 1, "modes_x": 3, "modes_y": 3, "modes_t": 2, "grid_size": 8}
    )
    return result


def make_smoke_data(config: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    data_cfg = config["data"]
    n = int(data_cfg["train_resolution"])
    input_steps = int(data_cfg["input_steps"])
    output_steps = int(data_cfg["output_steps"])
    total_steps = input_steps + output_steps
    samples = int(data_cfg["train_samples"]) + int(data_cfg["validation_samples"])
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    trajectory = torch.randn(samples, n, n, total_steps, generator=generator) * 0.05
    x = trajectory[..., :input_steps]
    y = trajectory[..., input_steps:]
    ntrain = int(data_cfg["train_samples"])
    return x[:ntrain], y[:ntrain], x[ntrain:], y[ntrain:]


def load_training_data(config: dict[str, Any], smoke: bool = False) -> tuple[torch.Tensor, ...]:
    if smoke:
        return make_smoke_data(config)
    data_cfg = config["data"]
    path = resolve_path(data_cfg["train_file"])
    x, y = load_trajectory_file(
        path,
        field=str(data_cfg["field"]),
        input_steps=int(data_cfg["input_steps"]),
        output_steps=int(data_cfg["output_steps"]),
    )
    ntrain = int(data_cfg["train_samples"])
    nval = int(data_cfg["validation_samples"])
    if x.shape[0] < ntrain + nval:
        raise ValueError(f"expected at least {ntrain + nval} trajectories, got {x.shape[0]}")
    return x[:ntrain], y[:ntrain], x[-nval:], y[-nval:]


def load_test_data(config: dict[str, Any], smoke: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if smoke:
        _, _, x, y = make_smoke_data(config)
        return x[:1], y[:1]
    data_cfg = config["data"]
    path = resolve_path(data_cfg["test_file"])
    x, y = load_trajectory_file(
        path,
        field=str(data_cfg["field"]),
        input_steps=int(data_cfg["input_steps"]),
        output_steps=int(data_cfg["output_steps"]),
        start_step=int(data_cfg["test_start_step"]),
        dtype=torch.float64,
    )
    return x[-int(data_cfg["test_samples"]) :], y[-int(data_cfg["test_samples"]) :]


def output_path(config: dict[str, Any], key: str, output_dir: Path | None) -> Path:
    root = output_dir.resolve() if output_dir is not None else ROOT
    path = root / str(config["outputs"][key])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def train_base(
    config: dict[str, Any],
    model: SFNO,
    train_data: tuple[torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    training = config["training"]
    loader = DataLoader(
        TensorDataset(*train_data),
        batch_size=int(training["batch_size"]),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(training["learning_rate"]),
        div_factor=float(training.get("scheduler_div_factor", 1000.0)),
        final_div_factor=float(training.get("scheduler_final_div_factor", 10000.0)),
        steps_per_epoch=len(loader),
        epochs=int(training["epochs"]),
    )
    train_loss = float("nan")
    model.train()
    for epoch in range(int(training["epochs"])):
        total = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, out_steps=targets.shape[-1])
            loss = relative_l2(prediction, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach())
        train_loss = total / len(loader)
        print(f"epoch={epoch + 1} train_loss={train_loss:.8e}")

    model.eval()
    with torch.no_grad():
        val_x, val_y = (tensor.to(device) for tensor in val_data)
        val_loss = float(relative_l2(model(val_x, out_steps=val_y.shape[-1]), val_y))
    print(f"val_loss={val_loss:.8e}")
    return train_loss, val_loss


def build_refiner(
    config: dict[str, Any],
    model: SFNO,
    batch_size: int,
    n_grid: int,
    dtype: torch.dtype,
    device: torch.device,
) -> OutConvFT:
    model_cfg = config["model"]
    fine = config["fine_tuning"]
    refiner = OutConvFT(
        int(model_cfg["modes_x"]),
        int(model_cfg["modes_y"]),
        int(model_cfg["modes_t"]),
        batch_size=batch_size,
        n_grid=n_grid,
        out_steps=int(config["data"]["output_steps"]),
        dt=float(fine["dt"]),
        delta=float(fine["delta"]),
        diam=float(fine["diameter"]),
        visc=float(fine["viscosity"]),
        bdf_weight=tuple(float(value) for value in fine["bdf_weight"]),
        dtype=dtype,
    )
    refiner._update_spectral_conv_weights(
        int(fine["modes_x"]),
        int(fine["modes_y"]),
        int(fine["modes_t"]),
        device=device,
        model=model.output_operator,
    )
    return refiner.to(device=device, dtype=dtype)


def refine_sample(
    config: dict[str, Any],
    model: SFNO,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[OutConvFT, dict[str, float]]:
    fine = config["fine_tuning"]
    inputs, targets = inputs.to(device), targets.to(device)
    model = model.to(device=device, dtype=inputs.dtype).eval()
    model.add_latent_hook("reduction")
    with torch.no_grad():
        base_prediction = model(inputs, out_steps=targets.shape[-1])
        latent = model.latent_tensors["reduction"].detach()

    refiner = build_refiner(config, model, inputs.shape[0], inputs.shape[1], inputs.dtype, device)
    residual_loss = SobolevLoss(
        n_grid=inputs.shape[1],
        relative=False,
        time_average=True,
        norm_order=float(fine["norm_order"]),
        alpha=float(fine["alpha"]),
        freq_cutoff=inputs.shape[1] // 2 + 1,
    ).to(device=device, dtype=inputs.dtype)
    optimizer = torch.optim.Adam(
        [
            {"params": refiner.conv.bias, "lr": float(fine["bias_learning_rate"])},
            {"params": refiner.conv.weight, "lr": float(fine["weight_learning_rate"])},
        ]
    )
    forcing = forcing_field(inputs.shape[1], device, inputs.dtype).unsqueeze(0)
    if inputs.shape[0] > 1:
        forcing = forcing.expand(inputs.shape[0], -1, -1)

    refined: dict[str, torch.Tensor] | None = None
    for step in range(int(fine["iterations"])):
        optimizer.zero_grad(set_to_none=True)
        refined = refiner(latent, inputs, forcing, out_steps=targets.shape[-1])
        loss = residual_loss(refined["residual"])
        loss.backward()
        optimizer.step()
        print(f"refine_step={step + 1} residual_hminus1={float(loss.detach()):.8e}")

    assert refined is not None
    with torch.no_grad():
        refined = refiner(latent, inputs, forcing, out_steps=targets.shape[-1])
        metrics = {
            "base_relative_l2_error": float(relative_l2(base_prediction, targets)),
            "relative_l2_error": float(relative_l2(refined["w"], targets)),
            "residual_hminus1": float(residual_loss(refined["residual"])),
        }
    return refiner, metrics


def run(config: dict[str, Any], smoke: bool, output_dir: Path | None, device: torch.device) -> dict[str, float]:
    active = smoke_config(config) if smoke else copy.deepcopy(config)
    torch.manual_seed(int(active["experiment"]["seed"]))
    train_x, train_y, val_x, val_y = load_training_data(active, smoke=smoke)
    model = build_model(active).to(device)
    train_loss, val_loss = train_base(active, model, (train_x, train_y), (val_x, val_y), device)
    torch.save(model.state_dict(), output_path(active, "base_checkpoint", output_dir))

    test_x, test_y = load_test_data(active, smoke=smoke)
    if not smoke:
        index = int(active["fine_tuning"].get("sample_index", 0)) % test_x.shape[0]
        test_x, test_y = test_x[index : index + 1], test_y[index : index + 1]
    refiner, metrics = refine_sample(active, model, test_x, test_y, device)
    torch.save(refiner.state_dict(), output_path(active, "refined_checkpoint", output_dir))
    metrics.update({"train_loss": train_loss, "val_loss": val_loss, "smoke": bool(smoke)})
    metrics_path = output_path(active, "metrics", output_dir)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"relative_l2_error={metrics['relative_l2_error']:.8e}")
    print(f"residual_hminus1={metrics['residual_hminus1']:.8e}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run(config, smoke=args.smoke, output_dir=args.output_dir, device=torch.device(args.device))


if __name__ == "__main__":
    main()
