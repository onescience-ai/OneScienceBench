import argparse
import json
import random

import numpy as np
import torch

from dataset import create_dataloaders
from metrics import GridMSEAccumulator, compute_sample_normalized_grid_mse
from utils import build_model, load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train partially shared-decoder DeepCFD with Total_MSE."
    )
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def loss_func(output, target):
    """Training objective exactly aligned with sample/grid-mean Total_MSE."""
    return compute_sample_normalized_grid_mse(output, target)["Total_MSE"]


def evaluate(model, loader, device):
    model.eval()
    accumulator = GridMSEAccumulator(device)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            accumulator.update(model(inputs), targets)
    return accumulator.compute()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    weighted_loss_sum = 0.0
    sample_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(inputs.shape[0])
        weighted_loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Training loader is empty")
    return weighted_loss_sum / sample_count


def main():
    args = parse_args()
    config, config_path = load_config(args.config)
    data_config = dict(config["data"])
    data_config["directory"] = resolve_path(data_config["directory"])
    training_config = config["training"]

    set_seed(data_config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, _ = create_dataloaders(**data_config)

    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    output_dir = resolve_path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / training_config["checkpoint_name"]
    metrics_path = output_dir / training_config.get(
        "metrics_filename", "train_metrics.jsonl"
    )
    metrics_path.write_text("", encoding="utf-8")

    print(f"Config: {config_path}")
    print(f"Data: {data_config['directory']}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        "Decoder sharing: "
        f"{model.shared_decoder_stages}/{model.num_stages} deepest stages"
    )
    print("Selection metric: Total_MSE (sample/grid mean, channel sum)")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(training_config["num_epochs"]):
        train_total_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        if (epoch + 1) % training_config["eval_interval"] != 0:
            continue

        val_metrics = evaluate(model, test_loader, device)
        val_loss = val_metrics["Total_MSE"]
        record = {
            "epoch": epoch + 1,
            "train_Total_MSE": train_total_mse,
            "val_loss": val_loss,
            **val_metrics,
        }
        with open(metrics_path, "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"Epoch {epoch + 1} | train_Total_MSE={train_total_mse:.6e} "
            f"| val_Total_MSE={val_loss:.6e} "
            f"| Ux={val_metrics['Ux_MSE']:.6e} "
            f"Uy={val_metrics['Uy_MSE']:.6e} "
            f"p={val_metrics['p_MSE']:.6e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": dict(config["model"]),
                    "epoch": epoch + 1,
                    "selection_metric": "Total_MSE",
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= training_config["patience"]:
                print("Early stopping")
                break


if __name__ == "__main__":
    main()
