import os
import sys
import time
import yaml
from pathlib import Path

import airfrans as af
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import AirfransFlowDataset, subsample_dataset
from mario import create_inr_instance, load_inr
from mario import training_step


def _find_latents(cfg):
    train_path = cfg.dataset.train_latents_path
    test_path = cfg.dataset.test_latents_path
    if train_path and test_path:
        return Path(train_path), Path(test_path)
    if not cfg.dataset.auto_find_latents:
        raise FileNotFoundError(
            "Set dataset.train_latents_path/test_latents_path or enable auto_find_latents"
        )

    root = Path(__file__).resolve().parent / "trainings"
    pattern = f"training_sdf_*/modulations/{cfg.dataset.task}_train_{cfg.dataset.geom_latent_dim}.npz"
    candidates = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            "No SDF latents found. Run `python train_sdf.py` first."
        )
    train_path = candidates[-1]
    test_path = train_path.with_name(
        f"{cfg.dataset.task}_test_{cfg.dataset.geom_latent_dim}.npz"
    )
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    print(f"Using geometry latents: {train_path.parent}")
    return train_path, test_path


@hydra.main(version_base=None, config_path=".", config_name="config_out")
def main(cfg: DictConfig) -> None:
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_latent_path, test_latent_path = _find_latents(cfg)
    train_data, train_names = af.dataset.load(
        root=cfg.dataset.root_path, task=cfg.dataset.task, train=True
    )
    test_data, test_names = af.dataset.load(
        root=cfg.dataset.root_path, task=cfg.dataset.task, train=False
    )
    train_latents = np.load(train_latent_path)
    test_latents = np.load(test_latent_path)

    train_dataset = AirfransFlowDataset(
        train_data,
        train_names,
        geom_latents=train_latents["train_modulations"],
        mode="train",
    )
    coef_norm = train_dataset.coef_norm
    test_dataset = AirfransFlowDataset(
        test_data,
        test_names,
        geom_latents=test_latents["val_modulations"],
        mode="test",
        coef_norm=coef_norm,
    )
    val_dataset = test_dataset
    ntrain, nval = len(train_dataset), len(val_dataset)

    inr_model = create_inr_instance(
        cfg, input_dim=cfg.inr.in_dim, output_dim=cfg.inr.out_dim, device=device
    )
    optimizer = torch.optim.AdamW(
        inr_model.parameters(), lr=cfg.optim.lr_inr, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-5
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = (
        Path(__file__).resolve().parent
        / "trainings"
        / f"training_result_{timestamp}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg.dataset.train_latents_path = str(train_latent_path)
    cfg.dataset.test_latents_path = str(test_latent_path)
    with open(results_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)
    torch.save(coef_norm, results_dir / "normalization.pt")

    run_name = (
        f"{cfg.inr.model_type}_nl{cfg.inr.depth}_np{cfg.optim.num_points}"
        f"_epochs{cfg.optim.epochs}_lr{cfg.optim.lr_inr}_ldim{cfg.inr.latent_dim}"
    )
    ckpt_path = results_dir / f"{run_name}.pt"
    best_loss = float("inf")
    train_history, val_history = [], []

    start_epoch = 0
    if cfg.restart_training and cfg.saved_model_path:
        checkpoint = torch.load(cfg.saved_model_path, map_location=device)
        inr_model.load_state_dict(checkpoint["inr_in"])
        optimizer.load_state_dict(checkpoint["optimizer_inr_in"])
        start_epoch = checkpoint["epoch"] + 1

    predict_scalars = cfg.inr.out_scalar_dim > 0
    for epoch in tqdm(range(start_epoch, cfg.optim.epochs), desc="Flow epoch"):
        inr_model.train()
        train_total = 0.0
        train_field = 0.0
        train_loader = DataLoader(
            subsample_dataset(train_dataset, cfg.optim.num_points),
            batch_size=cfg.optim.batch_size,
            shuffle=True,
        )
        for graph in train_loader:
            graph = graph.to(device)
            outputs = training_step(
                inr_model,
                graph,
                latent_cond=True,
                predict_scalars=predict_scalars,
            )
            optimizer.zero_grad()
            outputs["loss"].backward()
            nn.utils.clip_grad_value_(inr_model.parameters(), clip_value=1.0)
            optimizer.step()
            n_graphs = graph.num_graphs
            train_total += outputs["loss"].detach().item() * n_graphs
            train_field += outputs["field_loss"].detach().item() * n_graphs

        train_loss = train_total / ntrain
        train_field_loss = train_field / ntrain
        train_history.append(train_loss)

        inr_model.eval()
        val_total = 0.0
        val_field = 0.0
        comp_loss = np.zeros(cfg.inr.out_dim, dtype=np.float64)
        val_loader = DataLoader(
            subsample_dataset(val_dataset, cfg.optim.num_points),
            batch_size=cfg.optim.batch_size_val or cfg.optim.batch_size,
            shuffle=False,
        )
        for graph in val_loader:
            graph = graph.to(device)
            outputs = training_step(
                inr_model,
                graph,
                latent_cond=True,
                predict_scalars=predict_scalars,
            )
            n_graphs = graph.num_graphs
            val_total += outputs["loss"].detach().item() * n_graphs
            val_field += outputs["field_loss"].detach().item() * n_graphs
            comp_loss += outputs["field_loss_components"].cpu().numpy() * n_graphs

        val_loss = val_total / nval
        val_field_loss = val_field / nval
        val_comp = comp_loss / nval
        val_history.append(val_loss)
        scheduler.step(val_loss)
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_field_loss": train_field_loss,
                "val_loss": val_loss,
                "val_field_loss": val_field_loss,
                **{f"val_component_{i}": v for i, v in enumerate(val_comp)},
            }
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch,
                    "inr_in": inr_model.state_dict(),
                    "optimizer_inr_in": optimizer.state_dict(),
                },
                ckpt_path,
            )

        if epoch % 10 == 0 or epoch == cfg.optim.epochs - 1:
            plt.figure()
            plt.plot(train_history, label="train")
            plt.plot(val_history, label="val")
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "loss.png", dpi=140)
            plt.close()

    # Save unnormalized test predictions beside the checkpoint.
    best = torch.load(ckpt_path, map_location=device)
    inr_model = load_inr(
        best["inr_in"], cfg, cfg.inr.in_dim, cfg.inr.out_dim, device=device
    )
    output_fields = [
        "Velocity-x",
        "Velocity-y",
        "Pressure",
        "Turbulent-viscosity",
    ]
    out_mean = np.array([coef_norm["output"]["mean"][f] for f in output_fields])
    out_std = np.array([coef_norm["output"]["std"][f] for f in output_fields])
    preds = []
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(device)
            field_n = inr_model.modulated_forward(
                graph.input, graph.cond[graph.batch]
            ).cpu().numpy()
            preds.append(field_n * out_std + out_mean)
    torch.save({"fields_pred": preds}, results_dir / "predictions.pt")
    print(f"Training outputs saved to: {results_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
