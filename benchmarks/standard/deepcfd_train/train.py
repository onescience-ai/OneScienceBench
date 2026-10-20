import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config_utils import load_config, resolve_path
from dataset import create_dataloaders
from utils import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepCFD.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_func(output, target, weights):
    channel_losses = torch.stack(
        [
            (output[:, 0] - target[:, 0]) ** 2,
            (output[:, 1] - target[:, 1]) ** 2,
            torch.abs(output[:, 2] - target[:, 2]),
        ],
        dim=1,
    )
    return torch.sum(channel_losses / weights)


def evaluate(model, loader, device, weights):
    model.eval()
    total_loss = 0.0
    channel_errors = torch.zeros(3, dtype=torch.float64, device=device)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            total_loss += loss_func(output, targets, weights).item()
            channel_errors += torch.sum(
                (output - targets) ** 2,
                dim=(0, 2, 3),
                dtype=torch.float64,
            )

    if len(loader) == 0:
        raise RuntimeError("Evaluation loader is empty")
    return total_loss / len(loader), channel_errors.cpu().tolist()


def main():
    args = parse_args()
    config, config_path = load_config(args.config)
    data_config = dict(config["data"])
    data_config["directory"] = resolve_path(data_config["directory"])
    training_config = config["training"]

    set_seed(data_config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, loss_weights = create_dataloaders(**data_config)
    loss_weights = loss_weights.to(device)

    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    output_dir = resolve_path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / training_config["checkpoint_name"]

    print(f"Config: {config_path}")
    print(f"Data: {data_config['directory']}")
    print(f"Device: {device}")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(training_config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss = loss_func(output, targets, loss_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if len(train_loader) == 0:
            raise RuntimeError("Training loader is empty")

        if (epoch + 1) % training_config["eval_interval"] != 0:
            continue

        val_loss, channel_errors = evaluate(
            model,
            test_loader,
            device,
            loss_weights,
        )
        average_train_loss = train_loss / len(train_loader)
        print(
            f"Epoch {epoch} | train={average_train_loss:.4e} "
            f"val={val_loss:.4e} | Ux={channel_errors[0]:.2e} "
            f"Uy={channel_errors[1]:.2e} P={channel_errors[2]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": dict(config["model"]),
                    "epoch": epoch,
                    "val_loss": val_loss,
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
