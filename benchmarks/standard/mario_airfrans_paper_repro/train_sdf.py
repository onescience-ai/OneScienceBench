import os
import sys
import time
import yaml
from pathlib import Path

import airfrans as af
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SDFDataset, subsample_dataset
from mario import create_inr_instance, load_inr
from mario import graph_outer_step


@hydra.main(version_base=None, config_path=".", config_name="config_sdf")
def main(cfg: DictConfig) -> None:
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path(cfg.output_dir).expanduser().resolve()
    results_dir = base / "trainings" / f"training_sdf_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)

    train_data, train_names = af.dataset.load(
        root=cfg.dataset.root_path, task=cfg.dataset.task, train=True
    )
    test_data, test_names = af.dataset.load(
        root=cfg.dataset.root_path, task=cfg.dataset.task, train=False
    )
    train_dataset = SDFDataset(train_data, train_names, is_train=True)
    test_dataset = SDFDataset(
        test_data, test_names, is_train=False, coef_norm=train_dataset.coef_norm
    )
    train_ds = train_dataset.processed_dataset
    test_ds = test_dataset.processed_dataset
    n_train, n_test = len(train_ds), len(test_ds)

    inr_model = create_inr_instance(
        cfg,
        input_dim=cfg.inr.input_dim,
        output_dim=cfg.inr.output_dim,
        device=device,
    )
    alpha_in = nn.Parameter(torch.tensor([cfg.optim.lr_code], device=device))
    optimizer = torch.optim.AdamW(
        [
            {"params": inr_model.parameters()},
            {
                "params": alpha_in,
                "lr": cfg.optim.meta_lr_code,
                "weight_decay": 0,
            },
        ],
        lr=cfg.optim.lr_inr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.99, patience=200, min_lr=1e-5
    )

    best_val = float("inf")
    for epoch in tqdm(range(cfg.optim.epochs), desc="SDF epoch"):
        inr_model.train()
        train_loss = 0.0
        train_loader = DataLoader(
            subsample_dataset(train_ds, cfg.optim.num_points),
            batch_size=cfg.optim.batch_size,
            shuffle=True,
        )
        for batch in train_loader:
            batch = batch.to(device)
            batch.modulations = torch.zeros(
                batch.num_graphs, cfg.inr.latent_dim, device=device
            )
            out = graph_outer_step(
                inr_model,
                batch,
                cfg.optim.inner_steps,
                alpha_in,
                is_train=True,
            )
            optimizer.zero_grad()
            out["loss"].backward()
            nn.utils.clip_grad_value_(inr_model.parameters(), clip_value=1.0)
            optimizer.step()
            train_loss += out["loss"].item() * batch.num_graphs
        avg_train = train_loss / n_train
        wandb.log({"epoch": epoch, "train_loss": avg_train})

        if epoch % 5 == 0:
            # Gradients are required to fit the per-geometry modulations.
            inr_model.train()
            val_loss = 0.0
            val_loader = DataLoader(
                subsample_dataset(test_ds, cfg.optim.num_points),
                batch_size=cfg.optim.batch_size,
                shuffle=False,
            )
            for batch in val_loader:
                batch = batch.to(device)
                batch.modulations = torch.zeros(
                    batch.num_graphs, cfg.inr.latent_dim, device=device
                )
                out = graph_outer_step(
                    inr_model,
                    batch,
                    cfg.optim.inner_steps,
                    alpha_in,
                    is_train=False,
                )
                val_loss += out["loss"].item() * batch.num_graphs
            avg_val = val_loss / n_test
            wandb.log({"epoch": epoch, "val_loss": avg_val})
            scheduler.step(avg_val)
            if avg_val < best_val:
                best_val = avg_val
                torch.save(
                    {
                        "epoch": epoch,
                        "inr_in": inr_model.state_dict(),
                        "alpha_in": alpha_in.detach().cpu(),
                        "optimizer": optimizer.state_dict(),
                    },
                    results_dir / "best_inr.pt",
                )

    ckpt = torch.load(results_dir / "best_inr.pt", map_location=device)
    inr_model = load_inr(
        ckpt["inr_in"],
        cfg,
        cfg.inr.input_dim,
        cfg.inr.output_dim,
        device=device,
    )
    inr_model.eval()

    def fit_all_modulations(dataset):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        mods = []
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            batch.modulations = torch.zeros(
                batch.num_graphs, cfg.inr.latent_dim, device=device
            )
            out = graph_outer_step(
                inr_model,
                batch,
                cfg.optim.inner_steps,
                ckpt["alpha_in"].to(device),
                is_train=False,
            )
            total_loss += out["loss"].item() * batch.num_graphs
            mods.append(out["modulations"].detach().cpu().numpy())
        return np.concatenate(mods, axis=0), total_loss / len(dataset)

    train_mods, _ = fit_all_modulations(train_ds)
    test_mods, final_test_loss = fit_all_modulations(test_ds)
    print("Final SDF test loss:", final_test_loss)

    mod_dir = results_dir / "modulations"
    mod_dir.mkdir(exist_ok=True)
    np.savez(
        mod_dir / f"{cfg.dataset.task}_train_{cfg.inr.latent_dim}.npz",
        train_modulations=train_mods,
    )
    np.savez(
        mod_dir / f"{cfg.dataset.task}_test_{cfg.inr.latent_dim}.npz",
        val_modulations=test_mods,
    )
    print(f"Saved modulations under: {mod_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
