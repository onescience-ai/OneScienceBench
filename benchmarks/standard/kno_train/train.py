"""Optional standalone KNO training entrypoint matching the checkpoint schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import NavierStokesDatapipe
from inference import build_model, resolve_data_path
from utils import load_config, project_path, relative_l2, resolve_device, rollout, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent


def evaluate(model, loader, device, t_out: int, out_dim: int) -> tuple[float, float]:
    model.eval()
    squared_error = relative_error = 0.0
    element_count = sample_count = 0
    with torch.no_grad():
        for batch in loader:
            pos, state, target = (batch[key].to(device) for key in ("pos", "x", "y"))
            prediction = rollout(model, pos, state, t_out, out_dim)
            squared_error += F.mse_loss(prediction, target, reduction="sum").item()
            relative_error += relative_l2(prediction, target).sum().item()
            element_count += target.numel()
            sample_count += target.shape[0]
    return squared_error / element_count, relative_error / sample_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    if args.device:
        config["common"]["device"] = args.device
    resolve_data_path(config)
    seed_everything(int(config["common"].get("seed", 0)))
    device = resolve_device(str(config["common"]["device"]))
    datapipe = NavierStokesDatapipe(config["datapipe"])
    model_config = config["model"]
    model = build_model(model_config, datapipe.spatial_shape, config["datapipe"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    data = config["datapipe"]["data"]
    t_out, out_dim = int(data["t_out"]), int(data["out_dim"])
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else (
        project_path(config["training"]["weight_dir"], PROJECT_ROOT).resolve()
        / config["training"]["checkpoint_name"]
    )
    best_relative_l2 = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in datapipe.train_dataloader():
            pos, state, target = (batch[key].to(device) for key in ("pos", "x", "y"))
            optimizer.zero_grad(set_to_none=True)
            prediction = rollout(model, pos, state, t_out, out_dim, teacher_forcing=True, target=target)
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        scheduler.step()
        if epoch % args.eval_interval and epoch != args.epochs:
            continue
        validation_mse, validation_relative_l2 = evaluate(model, datapipe.test_dataloader(), device, t_out, out_dim)
        print(f"epoch={epoch} train_mse={total_loss / batch_count:.6e} val_mse={validation_mse:.6e} val_relative_l2={validation_relative_l2:.6e}")
        if validation_relative_l2 < best_relative_l2:
            best_relative_l2 = validation_relative_l2
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "datapipe_config": config["datapipe"],
                    "training_config": {"lr": args.lr, "weight_decay": args.weight_decay},
                    "spatial_shape": datapipe.spatial_shape,
                    "normalizer": datapipe.get_normalizer_state(),
                    "best_relative_l2": best_relative_l2,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
