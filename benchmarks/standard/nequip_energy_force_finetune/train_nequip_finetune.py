"""One-batch readout fine-tuning for the fixed Cu NequIP checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from portable_nequip.runtime import (
    build_energy_force_batch,
    load_model_from_checkpoint,
    predict_energy_forces,
)


def load_config(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def trainable_named_parameters(model):
    selected = []
    for name, parameter in model.named_parameters():
        trainable = "per_atom_energy_readout" in name
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError("NequIP readout parameter was not found")
    return selected


def build_model(config, device: str = "cpu"):
    model, _, _ = load_model_from_checkpoint(config["paths"]["checkpoint"], device)
    trainable_named_parameters(model)
    return model


def build_trainer(config, device: str = "cpu"):
    torch.manual_seed(int(config["training"]["seed"]))
    model = build_model(config, device)
    named = trainable_named_parameters(model)
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in named],
        lr=float(config["training"]["lr"]),
        eps=float(config["training"]["eps"]),
    )
    return model, optimizer


def train_step(model, optimizer, batch, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = predict_energy_forces(model, batch)
    num_atoms = float(batch["forces"].shape[0])
    energy_loss = torch.nn.functional.mse_loss(
        prediction["energy"] / num_atoms, batch["energy"] / num_atoms
    )
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
        "energy_mse_per_atom": float(energy_loss.detach()),
        "forces_mse": float(force_loss.detach()),
        "grad_norm": float(grad_norm.detach()),
    }


def train(config_path: str | Path, output_dir: str | Path, device: str = "cpu"):
    config = load_config(config_path)
    torch.manual_seed(int(config["training"]["seed"]))
    model, optimizer = build_trainer(config, device)
    batch = build_energy_force_batch(
        config["paths"]["data"], int(config["training"]["structure_index"]), device
    )
    history = []
    for _ in range(int(config["training"]["max_steps"])):
        history.append(train_step(model, optimizer, batch, config))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainable_names = [name for name, _ in trainable_named_parameters(model)]
    payload = {
        "format": "nequip-energy-force-finetune-v1",
        "model_name": "NequIPGNNModel",
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "completed_steps": len(history),
        "trainable_parameter_names": trainable_names,
        "source_checkpoint_sha256": "3ab4d7d891a36ed6346e661003fc30197d4b615171ccaec92cdd6b7bb1b4049b",
        "config": config,
        "history": history,
    }
    torch.save(payload, output / "nequip_finetuned.pt")
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
    parser.add_argument("--config", default=str(Path(__file__).with_name("nequip_finetune.yaml")))
    parser.add_argument("--output-dir", default="nequip_finetune_output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    return train(args.config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
