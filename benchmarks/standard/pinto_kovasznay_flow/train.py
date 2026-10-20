from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from data import build_kovasznay_batch
from model import PINTOModel


def load_config(path: str | Path = "config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict) -> PINTOModel:
    model_cfg = config["model"]
    data_cfg = config["data"]
    return PINTOModel(
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        activation=str(model_cfg["activation"]),
        domain_start=tuple(float(v) for v in data_cfg["domain_start"]),
        domain_stop=tuple(float(v) for v in data_cfg["domain_stop"]),
    )


def pde_residual_loss(model: PINTOModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    xy = batch["query_xy"].detach().clone().requires_grad_(True)
    pred = model(xy, batch["boundary_xy"], batch["boundary_values"])
    u, v, p = pred[..., 0:1], pred[..., 1:2], pred[..., 2:3]

    def grad(values: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(values.sum(), xy, create_graph=True, retain_graph=True)[0]

    gu = grad(u)
    gv = grad(v)
    gp = grad(p)
    ux, uy = gu[..., 0:1], gu[..., 1:2]
    vx, vy = gv[..., 0:1], gv[..., 1:2]
    px, py = gp[..., 0:1], gp[..., 1:2]
    uxx = grad(ux)[..., 0:1]
    uyy = grad(uy)[..., 1:2]
    vxx = grad(vx)[..., 0:1]
    vyy = grad(vy)[..., 1:2]
    nue = batch["nue"]
    fx = u * ux + v * uy + px - nue * (uxx + uyy)
    fy = u * vx + v * vy + py - nue * (vxx + vyy)
    div = ux + vy
    return torch.mean(fx**2) + torch.mean(fy**2) + torch.mean(div**2)


def train_one_step(config: dict, output_dir: Path) -> dict[str, float]:
    torch.manual_seed(int(config["experiment"]["seed"]))
    model = build_model(config)
    train_nue = config["data"]["train_nue"][: int(config["training"]["batch_size"])]
    batch = build_kovasznay_batch(
        train_nue,
        domain_samples=int(config["data"]["smoke_domain_samples"]),
        seq_len=int(config["data"]["smoke_seq_len"]),
        start=tuple(config["data"]["domain_start"]),
        stop=tuple(config["data"]["domain_stop"]),
        seed=int(config["experiment"]["seed"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    optimizer.zero_grad()
    pred = model(batch["query_xy"], batch["boundary_xy"], batch["boundary_values"])
    boundary_pred = model(batch["boundary_xy"], batch["boundary_xy"], batch["boundary_values"])
    data_loss = torch.mean((pred - batch["target"]) ** 2)
    boundary_loss = torch.mean((boundary_pred - batch["boundary_values"]) ** 2)
    residual_loss = pde_residual_loss(model, batch)
    total_loss = data_loss + float(config["training"]["boundary_loss_weight"]) * boundary_loss
    total_loss = total_loss + float(config["training"]["pde_loss_weight"]) * residual_loss
    total_loss.backward()
    optimizer.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / config["experiment"]["checkpoint_path"]
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_path)
    results_dir = output_dir / config["experiment"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "train_loss": float(total_loss.detach().item()),
        "data_loss": float(data_loss.detach().item()),
        "boundary_loss": float(boundary_loss.detach().item()),
        "residual_loss": float(residual_loss.detach().item()),
    }
    (results_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(
        "train_loss={train_loss:.8f} data_loss={data_loss:.8f} "
        "boundary_loss={boundary_loss:.8f} residual_loss={residual_loss:.8f}".format(**history)
    )
    return history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train_one_step(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
