"""Configuration and general utilities for the G-FNO NS experiment."""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    data_path: str = "./data/ns_V1e-4_N10000_T30.mat"
    T_in: int = 10
    T_out: int = 20
    ntrain: int = 1000
    nvalid: int = 100
    ntest: int = 100
    batch_size: int = 20
    num_workers: int = 0
    expected_resolution: int = 64
    # Official G-FNO NS uses no dataset normalization.
    normalization: str = "none"
    normalization_eps: float = 1e-6


@dataclass
class ModelConfig:
    num_channels: int = 1
    modes: int = 12
    width: int = 10


@dataclass
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    early_stopping: Optional[int] = 100


@dataclass
class RuntimeConfig:
    seed: int = 1
    device: str = "cuda"
    output_dir: str = "./results/ns_gfno2d_p4_seed1"
    verbose: bool = True
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def get_config() -> ExperimentConfig:
    """Return the default paper-oriented NS configuration."""

    return ExperimentConfig()


def config_to_dict(config: ExperimentConfig) -> dict:
    return asdict(config)


# -----------------------------------------------------------------------------
# Reproducibility and runtime
# -----------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available. Use --device cpu or change RuntimeConfig."
        )
    return torch.device(requested)


# -----------------------------------------------------------------------------
# Logging / files / checkpoints
# -----------------------------------------------------------------------------


def setup_logger(output_dir: str | Path, name: str = "gfno_ns") -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers when called from notebooks/tests repeatedly.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int,
    best_valid: float,
    config: Optional[ExperimentConfig] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_valid": best_valid,
        "model": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if config is not None:
        state["config"] = config_to_dict(config)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Also accept a plain state_dict for compatibility with older files.
        model.load_state_dict(checkpoint)
        return {"epoch": -1, "best_valid": float("inf")}

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------------------------------------------------------
# General metrics/evaluation
# -----------------------------------------------------------------------------


class LpLoss:
    """Relative Lp loss matching the official experiment's reduction."""

    def __init__(self, p: int = 2, size_average: bool = False):
        self.p = p
        self.size_average = size_average

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape or x.ndim != 3:
            raise ValueError(
                f"expected equal [batch, points, channels], got {x.shape}, {y.shape}"
            )
        diff = torch.norm(x - y, self.p, dim=1)
        denom = torch.norm(y, self.p, dim=1).clamp_min(1e-12)
        loss = (diff / denom).mean(dim=-1)
        return loss.mean() if self.size_average else loss.sum()


def autoregressive_rollout(
    model: torch.nn.Module,
    x: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Predict ``steps`` future frames with the rolling T_in history window."""

    preds = []
    for _ in range(steps):
        pred = model(x)
        preds.append(pred)
        x = torch.cat((x[..., 1:, :], pred), dim=-2)
    return torch.cat(preds, dim=-2)


def evaluate(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    steps: int,
    loss_fn: Optional[LpLoss] = None,
    normalizer: Optional[Any] = None,
) -> float:
    """Autoregressive validation/test metric.

    If a dataset normalizer is active, prediction and target are decoded before
    metric computation so the reported validation/test loss is in physical data
    space. With the official ``normalization='none'`` setting this is identical
    to the upstream behavior.
    """

    if loss_fn is None:
        loss_fn = LpLoss(size_average=False)

    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = autoregressive_rollout(model, x, steps)

            if normalizer is not None:
                pred_metric = normalizer.decode(pred)
                y_metric = normalizer.decode(y)
            else:
                pred_metric = pred
                y_metric = y

            total += loss_fn(
                pred_metric.reshape(len(pred_metric), -1, pred_metric.shape[-1]),
                y_metric.reshape(len(y_metric), -1, y_metric.shape[-1]),
            ).item()

    return total / len(loader.dataset)


def rotation_equivariance_error(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Mean relative error (%) under a 90-degree input/output rotation."""

    model.eval()
    with torch.no_grad():
        out = model(x)
        rotated_reference = out.rot90(k=1, dims=[1, 2])
        rotated_prediction = model(x.rot90(k=1, dims=[1, 2]))
        denom = rotated_reference.abs().clamp_min(1e-12)
        err = ((rotated_reference - rotated_prediction).abs() / denom).mean()
    return err.item() * 100.0


def count_parameters(model: torch.nn.Module) -> dict:
    return {
        "complex_count": sum(
            p.numel() * (1 + p.is_complex()) for p in model.parameters()
        ),
        "tensor_elements": sum(p.numel() for p in model.parameters()),
    }
