"""One-batch from-scratch energy-force training for the local TACE model.

Builds the model from the fixed architecture config with a deterministic seed,
runs a real one-structure energy+force forward, backpropagates, and performs a
single AdamW step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml

from tace_energy_force_train import (
    DATA_SHA256,
    build_model,
    build_train_batch,
    predict_energy_forces,
)


def load_config(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    arch = config.get("paths", {}).get("arch_config")
    if arch and not os.path.isabs(arch):
        config["paths"]["arch_config"] = str((path.parent / arch).resolve())
    return config


def trainable_named_parameters(model):
    """From-scratch training: every parameter is trainable."""
    return list(model.named_parameters())


def build_trainer(config, device: str = "cpu"):
    torch.manual_seed(int(config["training"]["seed"]))
    model = build_model(config, device)
    named = trainable_named_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    return model, optimizer


def train_step(model, optimizer, batch, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = predict_energy_forces(model, batch)
    energy_loss = torch.nn.functional.mse_loss(prediction["energy"], batch["energy"])
    force_loss = torch.nn.functional.mse_loss(prediction["forces"], batch["forces"])
    loss = (
        float(config["training"]["energy_weight"]) * energy_loss
        + float(config["training"]["force_weight"]) * force_loss
    )
    loss.backward()
    parameters = [parameter for _, parameter in trainable_named_parameters(model)]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, float(config["training"]["gradient_clip_norm"])
    )
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "energy_mse": float(energy_loss.detach()),
        "forces_mse": float(force_loss.detach()),
        "grad_norm": float(grad_norm.detach()),
    }


def train(config_path: str | Path, output_dir: str | Path, device: str = "cpu"):
    config = load_config(config_path)
    torch.manual_seed(int(config["training"]["seed"]))
    model, optimizer = build_trainer(config, device)
    batch = build_train_batch(
        config["paths"]["data"], int(config["training"]["structure_index"]), device
    )
    history = []
    for _ in range(int(config["training"]["max_steps"])):
        history.append(train_step(model, optimizer, batch, config))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainable_names = [name for name, _ in trainable_named_parameters(model)]
    payload = {
        "format": "tace-energy-force-train-v1",
        "model_name": "e3nnTACE",
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "completed_steps": len(history),
        "trainable_parameter_names": trainable_names,
        "data_sha256": DATA_SHA256,
        "config": config,
        "history": history,
    }
    torch.save(payload, output / "tace_trained.pt")
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
        encoding="utf-8",
    )
    (output / "train.log").write_text(
        f"completed_steps={len(history)}\n", encoding="utf-8"
    )
    return history[-1]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("tace_train.yaml")))
    parser.add_argument("--output-dir", default="tace_train_output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    return train(args.config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
