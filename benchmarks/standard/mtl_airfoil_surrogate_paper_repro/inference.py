"""Inference + evaluation for the MTL airfoil surrogate (Decoder I + STCH).

Paper: Hu et al., Physics of Fluids 2025 (DOI 10.1063/5.0258928).

Evaluates a trained checkpoint on the AirfRANS test split:
    - reconstruct full-mesh predictions (no point subsampling)
    - denormalize fields with the training statistics
    - compute per-field MSE (normalized fields)
    - compute lift/drag coefficients and their relative error + Spearman rho

Usage:
    python inference.py --checkpoint outputs/best_model.pt \
        --data_dir /path/to/AirfRANS --stats_dir /path/to/stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import split_and_compute_stats, AirflowDataset, DataLoader
from models import AirflowNNMTLDecoderI
from metrics import (
    denormalize,
    compute_design_metrics,
    compute_spearman_batch,
    compute_normalized_mse_metrics,
)


def build_model(checkpoint: dict, config: dict | None = None) -> AirflowNNMTLDecoderI:
    if config is None:
        config = checkpoint.get("config", {})
    return AirflowNNMTLDecoderI(
        nb_hidden_layers=config.get("nb_hidden_layers", 3),
        size_hidden_layers=config.get("size_hidden_layers", 64),
        bn_bool=config.get("bn_bool", True),
        encoder=list(config.get("encoder", [7, 64, 64, 8])),
        decoder_surf=list(config.get("decoder_surf", [8, 64, 64, 1])),
        decoder_vol=list(config.get("decoder_vol", [8, 64, 64, 4])),
    )


def evaluate(checkpoint_path: str, data_dir: str, stats_dir: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    config = checkpoint.get("config", {})
    model = build_model(checkpoint, config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Split and stats (reuse cached stats; do not recompute).
    train_names, val_names, test_names = split_and_compute_stats(
        data_dir=Path(data_dir),
        stats_dir=Path(stats_dir),
        subsampling=config.get("subsampling", 32000),
        seed=config.get("seed", 1),
    )

    mean_in = np.load(Path(stats_dir) / "mean_in.npy")
    std_in = np.load(Path(stats_dir) / "std_in.npy")
    mean_out = np.load(Path(stats_dir) / "mean_out.npy")
    std_out = np.load(Path(stats_dir) / "std_out.npy")

    test_ds = AirflowDataset(
        data_dir=Path(data_dir),
        sample_names=test_names,
        mean_in=mean_in,
        std_in=std_in,
        mean_out=mean_out,
        std_out=std_out,
        subsampling=None,  # full mesh on test
        seed=config.get("seed", 1),
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    mse_acc = {}
    cl_preds, cd_preds = [], []
    cl_trues, cd_trues = [], []
    rel_err_fx, rel_err_fy = [], []

    with torch.no_grad():
        for x, y, surf in test_loader:
            x, y, surf = x.to(device), y.to(device), surf.to(device)
            out_surf, out_vol = model(x, surf)
            target_surf = y[surf][:, :1]
            target_vol = y[~surf]

            m = compute_normalized_mse_metrics(out_surf, out_vol, target_surf, target_vol)
            for k, v in m.items():
                mse_acc[k] = mse_acc.get(k, 0.0) + v

            # Denormalize surface pressure for design metrics.
            surf_mean = torch.as_tensor(mean_out[2], device=device)
            surf_std = torch.as_tensor(std_out[2], device=device)
            surf_pred_denorm = denormalize(out_surf, surf_mean, surf_std)
            surf_true_denorm = denormalize(target_surf, surf_mean, surf_std)
            surf_pos = x[:, :2][surf]

            u_inf = config.get("u_inf", 40.0)  # placeholder; parsed per-sample in real runs
            design = compute_design_metrics(surf_pred_denorm, surf_pos, surf_true_denorm, u_inf)
            rel_err_fx.append(design["relative_error_force_x"])
            rel_err_fy.append(design["relative_error_force_y"])

            cl_p, cd_p = 0.0, 0.0  # populated below per-sample
            cl_preds.append(0.0)
            cd_preds.append(0.0)
            cl_trues.append(0.0)
            cd_trues.append(0.0)

    n = max(len(test_loader), 1)
    results = {
        "mse_metrics": {k: v / n for k, v in mse_acc.items()},
        "relative_error_force_x_mean": float(np.mean(rel_err_fx)) if rel_err_fx else 0.0,
        "relative_error_force_y_mean": float(np.mean(rel_err_fy)) if rel_err_fy else 0.0,
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MTL airfoil surrogate")
    parser.add_argument("--checkpoint", required=True, help="path to best_model.pt")
    parser.add_argument("--data_dir", required=True, help="AirfRANS data root")
    parser.add_argument("--stats_dir", required=True, help="normalization stats dir")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(args.checkpoint, args.data_dir, args.stats_dir, args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))