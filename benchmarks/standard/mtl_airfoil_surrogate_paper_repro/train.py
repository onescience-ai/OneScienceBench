"""Pure-torch training loop for the MTL airfoil surrogate (Decoder I + STCH).

Paper: Hu et al., Physics of Fluids 2025 (DOI 10.1063/5.0258928).

Training configuration (paper Section IV.B):
    - optimizer: Adam, lr = 1e-3
    - scheduler: one-cycle cosine, max lr = 1e-3
    - epochs: 800
    - batch size: 1 (one simulation per step), 32000 points subsampled
    - MTL weighting: STCH (mu=1, warmup_epoch=4), 2 tasks (surface / volume)

Usage:
    python train.py --data_dir /path/to/AirfRANS --stats_dir /path/to/stats \
        --epochs 800 --out outputs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dataset import split_and_compute_stats, make_loaders
from models import AirflowNNMTLDecoderI
from metrics import compute_normalized_mse_metrics
from weight_methods import make_weight_method


def train_epoch(model, loader, weight_method, optimizer, scheduler, device):
    model.train()
    epoch_loss_surf = 0.0
    epoch_loss_vol = 0.0
    steps = 0
    for x, y, surf in loader:
        x, y, surf = x.to(device), y.to(device), surf.to(device)
        optimizer.zero_grad()

        out_surf, out_vol = model(x, surf)
        target_surf = y[surf][:, :1]
        target_vol = y[~surf]
        loss_surf = torch.nn.functional.mse_loss(out_surf, target_surf)
        loss_vol = torch.nn.functional.mse_loss(out_vol, target_vol)

        combined = weight_method.backward(
            [loss_surf, loss_vol],
            shared_parameters=model.shared_parameters(),
            task_specific_parameters=[
                model.decoder_surf.parameters(),
                model.decoder_vol.parameters(),
            ],
            last_shared_parameters=model.last_shared_parameters(),
            optimizer=optimizer,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss_surf += loss_surf.item()
        epoch_loss_vol += loss_vol.item()
        steps += 1

    return epoch_loss_surf / max(steps, 1), epoch_loss_vol / max(steps, 1)


def validate(model, loader, device):
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for x, y, surf in loader:
            x, y, surf = x.to(device), y.to(device), surf.to(device)
            out_surf, out_vol = model(x, surf)
            target_surf = y[surf][:, :1]
            target_vol = y[~surf]
            loss_surf = torch.nn.functional.mse_loss(out_surf, target_surf)
            loss_vol = torch.nn.functional.mse_loss(out_vol, target_vol)
            total += (loss_surf + loss_vol).item()
            steps += 1
    return total / max(steps, 1)


def train(config: dict) -> dict:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(config["data_dir"])
    stats_dir = Path(config.get("stats_dir", str(data_dir / "stats")))

    train_names, val_names, test_names = split_and_compute_stats(
        data_dir=data_dir,
        stats_dir=stats_dir,
        train_name=config.get("train_name", "full_train"),
        test_name=config.get("test_name", "full_test"),
        val_split_ratio=config.get("val_split_ratio", 0.1),
        subsampling=config.get("subsampling", 32000),
        crop=config.get("crop"),
        seed=config.get("seed", 1),
    )

    train_loader, val_loader, test_loader = make_loaders(
        data_dir=data_dir,
        stats_dir=stats_dir,
        train_names=train_names,
        val_names=val_names,
        test_names=test_names,
        batch_size=config.get("batch_size", 1),
        num_workers=config.get("num_workers", 0),
        subsampling=config.get("subsampling", 32000),
        crop=config.get("crop"),
        seed=config.get("seed", 1),
    )

    model = AirflowNNMTLDecoderI(
        nb_hidden_layers=config.get("nb_hidden_layers", 3),
        size_hidden_layers=config.get("size_hidden_layers", 64),
        bn_bool=config.get("bn_bool", True),
        encoder=list(config.get("encoder", [7, 64, 64, 8])),
        decoder_surf=list(config.get("decoder_surf", [8, 64, 64, 1])),
        decoder_vol=list(config.get("decoder_vol", [8, 64, 64, 4])),
    ).to(device)

    weight_method = make_weight_method(
        config.get("weight_method", "stch"),
        mu=config.get("mu", 1.0),
        warmup_epoch=config.get("warmup_epoch", 4),
        n_tasks=config.get("n_tasks", 2),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    epochs = config.get("epochs", 800)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get("lr", 1e-3),
        total_steps=epochs * len(train_loader),
        pct_start=config.get("pct_start", 0.1),
        anneal_strategy="cos",
    )

    best_val = float("inf")
    best_state = None
    patience = config.get("early_stopping_patience", 200)
    no_improve = 0
    start = time.time()

    for epoch in range(epochs):
        ls, lv = train_epoch(
            model, train_loader, weight_method, optimizer, scheduler, device
        )
        val_loss = validate(model, val_loader, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:4d} | train_surf {ls:.4f} | train_vol {lv:.4f} | val {val_loss:.4f} | best {best_val:.4f}",
                flush=True,
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    # Restore best weights.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on the full test meshes (no subsampling).
    model.eval()
    test_metrics = {"val_loss": best_val, "best_epoch": epoch, "time_s": time.time() - start}
    with torch.no_grad():
        for x, y, surf in test_loader:
            x, y, surf = x.to(device), y.to(device), surf.to(device)
            out_surf, out_vol = model(x, surf)
            target_surf = y[surf][:, :1]
            target_vol = y[~surf]
            m = compute_normalized_mse_metrics(
                out_surf, out_vol, target_surf, target_vol
            )
            for k, v in m.items():
                test_metrics[k] = test_metrics.get(k, 0.0) + v
        # average over samples
        n_test = max(len(test_loader), 1)
        for k in ("surface_p", "volume_p", "volume_u", "volume_v", "volume_nut", "volume_all"):
            if k in test_metrics:
                test_metrics[k] = test_metrics[k] / n_test

    # Save model and metrics.
    out_dir = Path(config.get("out", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
        out_dir / "best_model.pt",
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, indent=2, ensure_ascii=False)

    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train MTL airfoil surrogate")
    parser.add_argument("--data_dir", required=True, help="AirfRANS data root with manifest.json")
    parser.add_argument("--stats_dir", default=None, help="dir to store/read normalization stats")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_method", default="stch")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    config["stats_dir"] = config["stats_dir"] or str(Path(args.data_dir) / "stats")
    result = train(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))