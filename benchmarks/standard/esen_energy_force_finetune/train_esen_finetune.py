from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from esen_energy_force_finetune import (
    build_finetune_batch,
    compute_energy_force_loss,
    load_checkpoint_payload,
    load_model,
    predict_energy_forces,
    trainable_named_parameters,
)


def load_config(path="esen_finetune.yaml"):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration must be a mapping")
    return config


def build_model(config, device="cpu"):
    loaded = load_model(config["checkpoint"], device=device, jd_path=config["jd_path"])
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    for name, parameter in model.named_parameters():
        if "denoising_linear" in name:
            parameter.requires_grad_(False)
    return model


def build_trainer(config, device="cpu"):
    model = build_model(config, device)
    parameters = [parameter for _, parameter in trainable_named_parameters(model)]
    optimizer = torch.optim.Adam(parameters, lr=float(config["training"]["lr"]), eps=float(config["training"]["eps"]))
    return model, optimizer


def train_step(model, optimizer, batch, payload, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = predict_energy_forces(model, batch)
    loss, components = compute_energy_force_loss(
        prediction, batch, payload,
        huber_delta=float(config["training"]["huber_delta"]),
        force_loss_ratio=float(config["training"]["force_loss_ratio"]),
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in trainable_named_parameters(model)],
        float(config["training"]["gradient_clip_norm"]),
    )
    optimizer.step()
    return {"loss": float(loss.detach()), "energy_loss": float(components["energy_loss"]), "force_loss": float(components["force_loss"]), "grad_norm": float(grad_norm)}


def train(config_path="esen_finetune.yaml", output_dir="outputs", device="cpu"):
    config = load_config(config_path)
    torch.manual_seed(int(config["training"]["seed"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint_payload(config["checkpoint"])
    model, optimizer = build_trainer(config, device)
    batch = build_finetune_batch(config["train_data"], config["training"]["structure_index"], device)
    before = {name: parameter.detach().clone() for name, parameter in trainable_named_parameters(model)}
    history = []
    for _ in range(int(config["training"]["max_steps"])):
        history.append(train_step(model, optimizer, batch, payload, config))
    changed = [name for name, parameter in trainable_named_parameters(model) if not torch.equal(before[name], parameter.detach())]
    record = {"step": int(config["training"]["max_steps"]), **history[-1], "changed_trainable_parameters": len(changed)}
    (output / "metrics.jsonl").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    (output / "train.log").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    artifact = {
        "format": "esen-energy-force-finetune-v1",
        "model_name": "esen_backbone_dens",
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_args": config["model"],
        "optimizer": optimizer.state_dict(),
        "completed_steps": int(config["training"]["max_steps"]),
        "trainable_parameter_names": [name for name, _ in trainable_named_parameters(model)],
        "config": config,
        "history": history,
    }
    torch.save(artifact, output / "esen_finetuned.pt")
    return {"output_dir": str(output), "history": history, "changed_trainable_parameters": changed}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="esen_finetune.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args.config, args.output_dir, device)


if __name__ == "__main__":
    main()
