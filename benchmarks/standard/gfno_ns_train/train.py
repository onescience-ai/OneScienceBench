"""Complete training/validation entry point for the G-FNO NS experiment."""

import argparse
from pathlib import Path
from timeit import default_timer

import torch
from tqdm import tqdm

from dataset import build_dataloaders
from model import GFNO
from utils import (
    LpLoss,
    config_to_dict,
    count_parameters,
    evaluate,
    get_config,
    load_checkpoint,
    resolve_device,
    rotation_equivariance_error,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GFNO2d-p4 on NS")

    # Common path/runtime overrides. Defaults live in utils.py.
    parser.add_argument("--data_path", default="/public/share/sugonhpcapp01/onestore/onedatasets/FNO_data/fno/data/ns_V1e-3_N5000_T50.mat")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)

    # Data overrides for debugging/ablation runs.
    parser.add_argument("--T_in", type=int, default=None)
    parser.add_argument("--T_out", type=int, default=None)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--nvalid", type=int, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--normalization",
        choices=["none", "unit_gaussian"],
        default=None,
    )

    # Model/training overrides.
    parser.add_argument("--modes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    return parser.parse_args()


def apply_cli_overrides(config, args: argparse.Namespace):
    if args.data_path is not None:
        config.data.data_path = args.data_path
    if args.output_dir is not None:
        config.runtime.output_dir = args.output_dir
    if args.device is not None:
        config.runtime.device = args.device
    if args.seed is not None:
        config.runtime.seed = args.seed

    for name in ("T_in", "T_out", "ntrain", "nvalid", "ntest", "batch_size"):
        value = getattr(args, name)
        if value is not None:
            setattr(config.data, name, value)
    if args.normalization is not None:
        config.data.normalization = args.normalization

    if args.modes is not None:
        config.model.modes = args.modes
    if args.width is not None:
        config.model.width = args.width
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.lr is not None:
        config.train.learning_rate = args.lr
    if args.weight_decay is not None:
        config.train.weight_decay = args.weight_decay
    return config


def build_scheduler(config, optimizer, train_loader):
    name = config.train.scheduler.lower()
    if name == "cosine":
        # Official G-FNO steps cosine scheduler after every optimizer update.
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs * len(train_loader),
        )
    if name == "none":
        return None
    raise ValueError(f"unsupported scheduler: {config.train.scheduler}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    *,
    verbose: bool,
) -> float:
    """Teacher-forcing one-step training over the expanded NS dataset."""

    model.train()
    total = 0.0
    iterator = tqdm(loader, disable=not verbose, leave=False)

    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(
            pred.reshape(len(pred), -1, pred.shape[-1]),
            y.reshape(len(y), -1, y.shape[-1]),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total += loss.item()
        if verbose:
            iterator.set_postfix(loss=f"{loss.item() / len(x):.5f}")

    return total / len(loader.dataset)


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(get_config(), args)

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    save_json(output_dir / "config.json", config_to_dict(config))

    set_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    logger.info("device=%s seed=%d", device, config.runtime.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data = build_dataloaders(
        data_path=config.data.data_path,
        T_in=config.data.T_in,
        T_out=config.data.T_out,
        ntrain=config.data.ntrain,
        nvalid=config.data.nvalid,
        ntest=config.data.ntest,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        expected_resolution=config.data.expected_resolution,
        normalization=config.data.normalization,
        normalization_eps=config.data.normalization_eps,
        pin_memory=config.runtime.pin_memory and device.type == "cuda",
    )
    logger.info(
        "trajectory split train=%s valid=%s test=%s",
        tuple(data.train_trajectories.shape),
        tuple(data.valid_trajectories.shape),
        tuple(data.test_trajectories.shape),
    )
    logger.info(
        "teacher-forcing samples=%d normalization=%s",
        len(data.train_loader.dataset),
        config.data.normalization,
    )

    # ------------------------------------------------------------------
    # Model / optimizer / scheduler
    # ------------------------------------------------------------------
    model = GFNO(
        num_channels=config.model.num_channels,
        modes=config.model.modes,
        width=config.model.width,
        initial_step=config.data.T_in,
    ).to(device)
    param_info = count_parameters(model)
    logger.info("model=GFNO2d parameters=%s", param_info)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(config, optimizer, data.train_loader)
    loss_fn = LpLoss(size_average=False)

    start_epoch = 0
    best_valid = float("inf")
    best_epoch = -1
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_valid = float(checkpoint.get("best_valid", float("inf")))
        best_epoch = int(checkpoint.get("epoch", -1))
        logger.info("resumed checkpoint=%s start_epoch=%d", args.resume, start_epoch)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    # Seed the local best checkpoint when resuming into a new output directory.
    if args.resume is not None and not best_path.exists():
        save_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=max(start_epoch - 1, 0),
            best_valid=best_valid,
            config=config,
        )
    history = []
    train_start = default_timer()

    # ------------------------------------------------------------------
    # Complete train + autoregressive validation loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, config.train.epochs):
        epoch_start = default_timer()
        train_loss = train_one_epoch(
            model,
            data.train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            verbose=config.runtime.verbose,
        )

        valid_loss = evaluate(
            model,
            data.valid_loader,
            device=device,
            steps=config.data.T_out,
            loss_fn=loss_fn,
            normalizer=data.normalizer,
        )
        lr = optimizer.param_groups[0]["lr"]
        elapsed = default_timer() - epoch_start

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "lr": lr,
            "seconds": elapsed,
        }
        history.append(record)
        logger.info(
            "epoch=%03d train=%.6f valid=%.6f lr=%.3e time=%.2fs",
            epoch + 1,
            train_loss,
            valid_loss,
            lr,
            elapsed,
        )

        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_valid=min(best_valid, valid_loss),
            config=config,
        )

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_valid=best_valid,
                config=config,
            )
            logger.info("saved new best checkpoint -> %s", best_path)

        patience = config.train.early_stopping
        if patience is not None and epoch - best_epoch > patience:
            logger.info("early stopping at epoch=%d", epoch + 1)
            break

    train_seconds = default_timer() - train_start
    save_json(output_dir / "history.json", history)

    # ------------------------------------------------------------------
    # Final best-checkpoint validation/test and equivariance check
    # ------------------------------------------------------------------
    load_checkpoint(best_path, model=model, device=device)
    final_valid = evaluate(
        model,
        data.valid_loader,
        device=device,
        steps=config.data.T_out,
        loss_fn=loss_fn,
        normalizer=data.normalizer,
    )
    test_loss = evaluate(
        model,
        data.test_loader,
        device=device,
        steps=config.data.T_out,
        loss_fn=loss_fn,
        normalizer=data.normalizer,
    )

    x0, _ = next(iter(data.test_loader))
    x0 = x0.to(device, non_blocking=True)
    rotation_error = rotation_equivariance_error(model, x0)

    summary = {
        "model_class": "xx",
        "model": "GFNO2d-p4",
        "best_epoch": best_epoch + 1,
        "best_valid": best_valid,
        "final_valid": final_valid,
        "test_loss": test_loss,
        "rotation_equivariance_percent": rotation_error,
        "train_seconds": train_seconds,
        "parameters": param_info,
        "best_checkpoint": str(best_path),
        "normalization": config.data.normalization,
        "official_split": {
            "train": "first ntrain trajectories",
            "valid": "block immediately before final test trajectories",
            "test": "final ntest trajectories",
        },
    }
    save_json(output_dir / "summary.json", summary)

    logger.info("best_epoch=%d best_valid=%.6f", best_epoch + 1, best_valid)
    logger.info("final_valid=%.6f test=%.6f", final_valid, test_loss)
    logger.info("rotation_equivariance_error=%.6g%%", rotation_error)
    logger.info("best_checkpoint=%s", best_path)


if __name__ == "__main__":
    main()
