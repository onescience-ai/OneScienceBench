"""One-batch NequIP energy-force training from deterministic initialization."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from portable_nequip.model import build_nequip_model
from portable_nequip.runtime import build_energy_force_batch, predict_energy_forces


def load_config(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config, device: str = "cpu"):
    seed = int(config["training"]["seed"])
    return build_nequip_model(seed=seed).to(device)


def build_trainer(config, device: str = "cpu"):
    torch.manual_seed(int(config["training"]["seed"]))
    model = build_model(config, device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
    energy_weight = float(config["training"]["energy_weight"])
    force_weight = float(config["training"]["force_weight"])
    loss = energy_weight * energy_loss + force_weight * force_loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["training"]["gradient_clip_norm"])
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
    payload = {
        "format": "nequip-energy-force-train-v1",
        "model_name": "NequIPGNNModel",
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "completed_steps": len(history),
        "config": config,
        "history": history,
    }
    torch.save(payload, output / "nequip_trained.pt")
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
    parser.add_argument("--config", default=str(Path(__file__).with_name("nequip_train.yaml")))
    parser.add_argument("--output-dir", default="nequip_train_output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    return train(args.config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
