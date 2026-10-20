"""Training entry point for the flattened 2-D INFINITY reproduction.

Stages
------
1. Train six field-specific modulated INRs with second-order CAVIA updates.
2. Re-encode every train/test function in three gradient steps.
3. Standardize latent codes using train statistics and fit g_psi in code space.

Examples
--------
python train.py --config config.yaml --stage inr --field sdf
python train.py --config config.yaml --stage inr --field vx
python train.py --config config.yaml --stage processor
python train.py --config config.yaml --stage all
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datset import (
    AirfRANSDataset,
    NormalizationStats,
    compute_normalization_stats,
    field_training_tensors,
    load_full_split,
)
from model import ALL_FIELDS, PHYSICS_FIELDS, adapt_code, build_inr, build_processor
from utils import (
    checkpoint_dir,
    fit_code_stats,
    freeze_module,
    iter_index_batches,
    load_config,
    normalize_codes,
    resolve_device,
    set_seed,
)


def load_or_compute_data_stats(
    trainset: AirfRANSDataset,
    config: Mapping,
) -> NormalizationStats:
    path = checkpoint_dir(config) / "normalization.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return NormalizationStats.from_dict(payload)

    print("Computing global train-set normalization statistics ...")
    stats = compute_normalization_stats(trainset)
    torch.save(stats.to_dict(), path)
    return stats



def train_inr(
    field: str,
    trainset: AirfRANSDataset,
    stats: NormalizationStats,
    config: Mapping,
    device: torch.device,
) -> Path:
    """Train one modality-specific modulated INR with second-order meta-gradients."""
    if field not in ALL_FIELDS:
        raise KeyError(field)

    model = build_inr(field, config).to(device)
    opt_cfg = config["optimization"]["inr"][field]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.inr.parameters(), "lr": float(opt_cfg["outer_lr"])},
            {
                "params": [model.inner_lr],
                "lr": float(opt_cfg["inner_lr_meta_lr"]),
                "weight_decay": 0.0,
            },
        ],
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=500,
        threshold=0.01,
        threshold_mode="rel",
        min_lr=1e-5,
    )

    batch_size = int(opt_cfg["batch_size"])
    epochs = int(opt_cfg["epochs"])
    inner_steps = int(config["optimization"]["inner_steps"])
    n_points = int(config["data"]["sample_points_per_case"])
    use_sdf = bool(config["model"].get("physics_decoder_condition_on_sdf", False))
    clip = float(config["optimization"].get("gradient_clip_value", 1.0))
    log_every = int(config["runtime"].get("log_every", 100))
    ckpt_path = checkpoint_dir(config) / f"inr_{field}.pt"
    best_loss = float("inf")

    sampling_rng = torch.Generator().manual_seed(int(config["runtime"]["seed"]))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0

        # Gradient accumulation is case-by-case to avoid retaining 64 second-order
        # computation graphs at once; it is mathematically the same batch mean update.
        for batch in iter_index_batches(len(trainset), batch_size, sampling_rng):
            optimizer.zero_grad(set_to_none=True)
            batch_n = len(batch)
            for index in batch:
                case = trainset[index]
                coordinates, target = field_training_tensors(
                    case,
                    stats,
                    field,
                    n_points=n_points,
                    physics_decoder_condition_on_sdf=use_sdf,
                    generator=sampling_rng,
                )
                coordinates = coordinates.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                code = adapt_code(
                    model,
                    coordinates,
                    target,
                    inner_steps,
                    create_graph=True,
                )
                loss = F.mse_loss(model(coordinates, code), target)
                (loss / batch_n).backward()
                total_loss += float(loss.detach())
                seen += 1

            torch.nn.utils.clip_grad_value_(model.inr.parameters(), clip)
            optimizer.step()

        epoch_loss = total_loss / max(seen, 1)
        scheduler.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "field": field,
                    "epoch": epoch,
                    "train_mse": best_loss,
                    "state_dict": model.state_dict(),
                    "paper_inner_steps": inner_steps,
                    "physics_decoder_condition_on_sdf": use_sdf,
                },
                ckpt_path,
            )

        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
            print(
                f"[INR:{field}] epoch={epoch + 1:05d}/{epochs} "
                f"mse={epoch_loss:.6e} alpha={float(model.inner_lr.detach()):.6e}"
            )

    return ckpt_path


def load_trained_inr(field: str, config: Mapping, device: torch.device):
    path = checkpoint_dir(config) / f"inr_{field}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; train field {field!r} first.")
    model = build_inr(field, config).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return freeze_module(model)


def encode_dataset(
    dataset: AirfRANSDataset,
    models: Mapping[str, torch.nn.Module],
    stats: NormalizationStats,
    config: Mapping,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    """Auto-decode every modality into a code using the paper's three test steps."""
    codes = {field: [] for field in ALL_FIELDS}
    inlets = []
    n_points = int(config["data"]["sample_points_per_case"])
    steps = int(config["optimization"]["test_inner_steps"])
    use_sdf = bool(config["model"].get("physics_decoder_condition_on_sdf", False))
    sampling_rng = torch.Generator().manual_seed(int(config["runtime"]["seed"]))

    for index in range(len(dataset)):
        case = dataset[index]
        for field in ALL_FIELDS:
            coordinates, target = field_training_tensors(
                case,
                stats,
                field,
                n_points=n_points,
                physics_decoder_condition_on_sdf=use_sdf,
                generator=sampling_rng,
            )
            coordinates = coordinates.to(device)
            target = target.to(device)
            # We still need first derivatives with respect to the temporary code.
            with torch.enable_grad():
                code = adapt_code(
                    models[field],
                    coordinates,
                    target,
                    steps,
                    create_graph=False,
                )
            codes[field].append(code.detach().cpu())
        inlets.append(stats.normalized_inlet(case))

        if (index + 1) % 50 == 0 or index + 1 == len(dataset):
            print(f"Encoded {index + 1}/{len(dataset)} airfoils")

    stacked = {field: torch.stack(values, dim=0) for field, values in codes.items()}
    return stacked, torch.stack(inlets, dim=0)



def _processor_tensors(
    normalized_codes: Mapping[str, Tensor],
    normalized_inlet: Tensor,
) -> tuple[Tensor, Tensor]:
    x = torch.cat(
        (normalized_codes["sdf"], normalized_codes["n"], normalized_inlet),
        dim=-1,
    )
    y = torch.cat(tuple(normalized_codes[field] for field in PHYSICS_FIELDS), dim=-1)
    return x, y


def train_processor(
    trainset: AirfRANSDataset,
    testset: AirfRANSDataset,
    stats: NormalizationStats,
    config: Mapping,
    device: torch.device,
) -> Path:
    models = {field: load_trained_inr(field, config, device) for field in ALL_FIELDS}
    print("Encoding train functions into latent codes ...")
    train_codes, train_inlet = encode_dataset(trainset, models, stats, config, device)
    print("Encoding test functions into latent codes ...")
    test_codes, test_inlet = encode_dataset(testset, models, stats, config, device)

    code_stats = fit_code_stats(train_codes)
    train_norm = normalize_codes(train_codes, code_stats)
    test_norm = normalize_codes(test_codes, code_stats)
    x_train, y_train = _processor_tensors(train_norm, train_inlet)
    x_test, y_test = _processor_tensors(test_norm, test_inlet)

    proc_cfg = config["optimization"]["processor"]
    processor = build_processor(config).to(device)
    optimizer = torch.optim.AdamW(
        processor.parameters(),
        lr=float(proc_cfg["lr"]),
        weight_decay=float(proc_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, threshold=0.01, threshold_mode="rel", min_lr=5e-5
    )
    loader_generator = torch.Generator().manual_seed(int(config["runtime"]["seed"]))
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(proc_cfg["batch_size"]),
        shuffle=True,
        generator=loader_generator,
    )

    epochs = int(proc_cfg["epochs"])
    log_every = int(config["runtime"].get("log_every", 100))
    best_train = float("inf")
    processor_path = checkpoint_dir(config) / "processor.pt"

    for epoch in range(epochs):
        processor.train()
        train_sum = 0.0
        train_count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = processor(x_batch)
            loss = F.mse_loss(prediction, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * x_batch.shape[0]
            train_count += x_batch.shape[0]

        processor.eval()
        with torch.no_grad():
            test_prediction = processor(x_test.to(device))
            test_loss = float(F.mse_loss(test_prediction, y_test.to(device)).cpu())
        train_loss = train_sum / max(train_count, 1)
        scheduler.step(train_loss)

        # The paper exposes only a train/test split, so do not select checkpoints
        # on the held-out 200-case test set. Test loss is diagnostic only.
        if train_loss < best_train:
            best_train = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "train_code_mse": train_loss,
                    "test_code_mse": test_loss,
                    "state_dict": processor.state_dict(),
                    "code_stats": code_stats,
                },
                processor_path,
            )

        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
            print(
                f"[processor] epoch={epoch + 1:04d}/{epochs} "
                f"train={train_loss:.6e} test={test_loss:.6e}"
            )

    # Build one self-contained inference bundle.
    best_processor = torch.load(processor_path, map_location="cpu", weights_only=False)
    bundle = {
        "config": copy.deepcopy(dict(config)),
        "data_stats": stats.to_dict(),
        "code_stats": best_processor["code_stats"],
        "inrs": {
            field: torch.load(
                checkpoint_dir(config) / f"inr_{field}.pt",
                map_location="cpu",
                weights_only=False,
            )["state_dict"]
            for field in ALL_FIELDS
        },
        "processor": best_processor["state_dict"],
    }
    bundle_path = checkpoint_dir(config) / "infinity_airfoil.pt"
    torch.save(bundle, bundle_path)
    print(f"Saved inference bundle: {bundle_path}")
    return bundle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage", choices=("inr", "processor", "all"), default="all")
    parser.add_argument("--field", choices=ALL_FIELDS, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["runtime"]["seed"]))
    device = resolve_device(config["runtime"].get("device", "cuda"))
    print(f"device={device}")

    trainset, testset = load_full_split(config)
    stats = load_or_compute_data_stats(trainset, config)

    if args.stage in ("inr", "all"):
        fields = (args.field,) if args.field else ALL_FIELDS
        for field in fields:
            train_inr(field, trainset, stats, config, device)

    if args.stage in ("processor", "all"):
        if args.stage == "all" and args.field is not None:
            raise ValueError("--field cannot be combined with --stage all because processor needs all six INRs.")
        train_processor(trainset, testset, stats, config, device)


if __name__ == "__main__":
    main()
