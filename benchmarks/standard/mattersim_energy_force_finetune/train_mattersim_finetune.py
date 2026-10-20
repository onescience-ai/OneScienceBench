"""Command-line and Python training entrypoint for MatterSim fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from mattersim_energy_force_finetune import (
    EXPECTED_MODEL_ARGS,
    MatterSimEnergyForceFineTuner,
    build_finetune_batch,
    load_pretrained_model,
)


ARTIFACT_FORMAT = "mattersim-1m-energy-force-finetune-v1"
CHECKPOINT_NAME = "mattersim_finetuned.pth"
METRICS_NAME = "metrics.jsonl"
LOG_NAME = "train.log"


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError("configuration must be a mapping")
    required = {"checkpoint", "train_data_path", "model", "training"}
    if not required.issubset(config):
        raise KeyError(f"configuration is missing {sorted(required - set(config))}")
    if dict(config["model"]) != EXPECTED_MODEL_ARGS:
        raise ValueError("configuration must use the fixed MatterSim 1M model arguments")
    return config


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(value)


def build_model(config: dict[str, Any], device: str | torch.device = "cpu"):
    return load_pretrained_model(config["checkpoint"], device=device)


def build_trainer(
    config: dict[str, Any], device: str | torch.device = "cpu"
) -> MatterSimEnergyForceFineTuner:
    training = config["training"]
    return MatterSimEnergyForceFineTuner(
        build_model(config, device=device),
        lr=float(training["lr"]),
        force_loss_ratio=float(training["force_loss_ratio"]),
        huber_delta=float(training["huber_delta"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )


def train_step(
    trainer: MatterSimEnergyForceFineTuner,
    batch: dict[str, Any],
) -> dict[str, float]:
    return trainer.train_step(batch)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def train(
    config: dict[str, Any],
    output_dir: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    torch.manual_seed(int(config["training"]["seed"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(config, device=device)
    batch = build_finetune_batch(
        config["train_data_path"],
        structure_index=int(config["training"]["structure_index"]),
        device=device,
    )
    history: list[dict[str, Any]] = []
    max_steps = int(config["training"]["max_steps"])
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    for step in range(max_steps):
        metrics = train_step(trainer, batch)
        history.append({"step": step + 1, **metrics})

    checkpoint_path = output / CHECKPOINT_NAME
    metrics_path = output / METRICS_NAME
    log_path = output / LOG_NAME
    payload = {
        "format": ARTIFACT_FORMAT,
        "model_name": "m3gnet",
        "model": trainer.model.state_dict(),
        "model_args": dict(EXPECTED_MODEL_ARGS),
        "optimizer": trainer.optimizer.state_dict(),
        "completed_steps": max_steps,
        "trainable_parameter_names": [
            name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
        ],
        "config": config,
        "history": history,
    }
    torch.save(payload, checkpoint_path)
    _write_jsonl(metrics_path, history)
    _write_jsonl(log_path, history)
    return {
        "model": trainer.model,
        "optimizer": trainer.optimizer,
        "history": history,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "log_path": str(log_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the local MatterSim 1M model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    result = train(config, args.output_dir, device=select_device(args.device))
    print(json.dumps({key: value for key, value in result.items() if key not in {"model", "optimizer"}}, indent=2))


if __name__ == "__main__":
    main()
