"""Inference and quantitative validation for a trained AirfRANS MARIO run.

Examples:
  python evaluate.py trainings/training_result_YYYYMMDD-HHMMSS
  python evaluate.py RUN_DIR --split test --sample-index 0
  python evaluate.py RUN_DIR --split test --save-predictions
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import airfrans as af
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import AirfransFlowDataset
from mario import load_inr

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_FIELDS = ["Velocity-x", "Velocity-y", "Pressure", "Turbulent-viscosity"]


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=map_location)


def load_run_config(run_dir):
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return OmegaConf.create(yaml.safe_load(f))


def find_checkpoint(run_dir, checkpoint=None):
    run_dir = Path(run_dir).resolve()
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    ignored = {"predictions.pt", "predictions_recomputed.pt", "normalization.pt"}
    for path in sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name in ignored:
            continue
        try:
            obj = safe_torch_load(path)
        except Exception:
            continue
        if isinstance(obj, dict) and "inr_in" in obj:
            return path
    raise FileNotFoundError(f"No MARIO flow checkpoint found under {run_dir}")


def resolve_latent_path(cfg, run_dir, split):
    key = "train_latents_path" if split == "train" else "test_latents_path"
    value = cfg.dataset.get(key)
    if value:
        path = Path(str(value)).expanduser()
        candidates = [path] if path.is_absolute() else [Path(run_dir) / path, PROJECT_ROOT / path, path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

    filename = f"{cfg.dataset.task}_{split}_{cfg.dataset.geom_latent_dim}.npz"
    candidates = sorted(
        (PROJECT_ROOT / "trainings").glob(f"training_sdf_*/modulations/{filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError(
        f"Could not find {filename}. Keep the matching training_sdf_* directory "
        "under ./trainings or update the run config."
    )


def build_datasets(run_dir):
    run_dir = Path(run_dir).resolve()
    cfg = load_run_config(run_dir)
    train_latent_path = resolve_latent_path(cfg, run_dir, "train")
    test_latent_path = resolve_latent_path(cfg, run_dir, "test")

    train_data, train_names = af.dataset.load(
        root=cfg.dataset.root_path, task=cfg.dataset.task, train=True
    )
    test_data, test_names = af.dataset.load(
        root=cfg.dataset.root_path, task=cfg.dataset.task, train=False
    )
    train_latents = np.load(train_latent_path)
    test_latents = np.load(test_latent_path)

    norm_path = run_dir / "normalization.pt"
    coef_norm = safe_torch_load(norm_path) if norm_path.exists() else None
    train_ds = AirfransFlowDataset(
        train_data,
        train_names,
        geom_latents=train_latents["train_modulations"],
        mode="train" if coef_norm is None else "test",
        coef_norm=coef_norm,
    )
    if coef_norm is None:
        coef_norm = train_ds.coef_norm
    test_ds = AirfransFlowDataset(
        test_data,
        test_names,
        geom_latents=test_latents["val_modulations"],
        mode="test",
        coef_norm=coef_norm,
    )
    return cfg, coef_norm, train_ds, test_ds


def load_model(run_dir, device=None, checkpoint=None):
    run_dir = Path(run_dir).resolve()
    cfg = load_run_config(run_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = find_checkpoint(run_dir, checkpoint)
    state = safe_torch_load(ckpt_path, map_location=device)
    model = load_inr(state["inr_in"], cfg, cfg.inr.in_dim, cfg.inr.out_dim, device=device)
    return model, device, ckpt_path


def output_mean_std(coef_norm):
    mean = np.asarray([coef_norm["output"]["mean"][f] for f in OUTPUT_FIELDS])
    std = np.asarray([coef_norm["output"]["std"][f] for f in OUTPUT_FIELDS])
    return mean, std


def predict_graph(model, graph, coef_norm, device):
    graph = graph.to(device)
    cond = graph.cond
    if cond.ndim == 1:
        cond = cond.unsqueeze(0)
    cond_per_point = cond[0].unsqueeze(0).expand(graph.input.shape[0], -1)
    with torch.inference_mode():
        pred_n = model.modulated_forward(graph.input, cond_per_point).cpu().numpy()
    target_n = graph.output.cpu().numpy()
    mean, std = output_mean_std(coef_norm)
    return pred_n, pred_n * std + mean, target_n, target_n * std + mean


def physical_xy(graph, coef_norm):
    xy_n = graph.input[:, :2].cpu().numpy()
    pmin = np.asarray(coef_norm["pos"]["min"])
    prange = np.asarray(coef_norm["pos"]["range"])
    return ((xy_n + 1.0) * 0.5) * prange + pmin


def scalar_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    err = pred - target
    mse = float(np.mean(err ** 2))
    denom = float(np.sum(target ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(err))),
        "relative_l2": math.sqrt(float(np.sum(err ** 2)) / denom) if denom > 0 else float("nan"),
        "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan"),
    }


def init_accumulator():
    return {
        f: {"n": 0, "sum_e2": 0.0, "sum_abs": 0.0, "sum_t2": 0.0, "sum_t": 0.0, "sum_t_sq": 0.0}
        for f in OUTPUT_FIELDS
    }


def update_accumulator(acc, pred, target, mask=None):
    if mask is not None:
        pred, target = pred[mask], target[mask]
    if pred.shape[0] == 0:
        return
    for j, field in enumerate(OUTPUT_FIELDS):
        p, t = pred[:, j].astype(np.float64), target[:, j].astype(np.float64)
        e = p - t
        a = acc[field]
        a["n"] += int(t.size)
        a["sum_e2"] += float(np.sum(e ** 2))
        a["sum_abs"] += float(np.sum(np.abs(e)))
        a["sum_t2"] += float(np.sum(t ** 2))
        a["sum_t"] += float(np.sum(t))
        a["sum_t_sq"] += float(np.sum(t ** 2))


def finalize_accumulator(acc):
    out = {}
    for field, a in acc.items():
        n = a["n"]
        if n == 0:
            out[field] = {k: float("nan") for k in ["mse", "rmse", "mae", "relative_l2", "r2"]}
            continue
        mse = a["sum_e2"] / n
        ss_tot = a["sum_t_sq"] - n * (a["sum_t"] / n) ** 2
        out[field] = {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": a["sum_abs"] / n,
            "relative_l2": math.sqrt(a["sum_e2"] / a["sum_t2"]) if a["sum_t2"] > 0 else float("nan"),
            "r2": 1.0 - a["sum_e2"] / ss_tot if ss_tot > 0 else float("nan"),
        }
    return out


def save_single_case(run_dir, split, idx, name, graph, pred_n, pred, target_n, target, coef_norm):
    out_dir = Path(run_dir) / "evaluation" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    out_path = out_dir / f"{split}_{idx:04d}_{safe_name}.npz"
    np.savez_compressed(
        out_path,
        case_name=np.asarray(name),
        split=np.asarray(split),
        xy=physical_xy(graph, coef_norm),
        prediction=pred,
        target=target,
        error=pred - target,
        prediction_normalized=pred_n,
        target_normalized=target_n,
        is_airfoil=(graph.is_airfoil.cpu().numpy() > 0.5),
        field_names=np.asarray(OUTPUT_FIELDS),
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="MARIO AirfRANS inference + validation")
    parser.add_argument("run_dir", help="trainings/training_result_<timestamp>")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-index", type=int, default=None, help="Evaluate one case instead of the whole split")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    _, coef_norm, train_ds, test_ds = build_datasets(run_dir)
    dataset = train_ds if args.split == "train" else test_ds
    model, device, ckpt_path = load_model(run_dir, args.device, args.checkpoint)

    if args.sample_index is not None:
        idx = args.sample_index if args.sample_index >= 0 else len(dataset) + args.sample_index
        if not 0 <= idx < len(dataset):
            raise IndexError(f"sample-index {args.sample_index} outside [0, {len(dataset)-1}]")
        graph = dataset[idx]
        pred_n, pred, target_n, target = predict_graph(model, graph, coef_norm, device)
        name = dataset.data_names[idx]
        out_path = save_single_case(run_dir, args.split, idx, name, graph, pred_n, pred, target_n, target, coef_norm)
        print(f"Checkpoint: {ckpt_path}\nCase: [{idx}] {name}\nSaved: {out_path}\n")
        print(f"{'field':24s} {'RMSE':>12s} {'MAE':>12s} {'rel-L2':>12s} {'R2':>12s}")
        for j, field in enumerate(OUTPUT_FIELDS):
            m = scalar_metrics(pred[:, j], target[:, j])
            print(f"{field:24s} {m['rmse']:12.5e} {m['mae']:12.5e} {m['relative_l2']:12.5e} {m['r2']:12.5f}")
        return

    n_cases = len(dataset) if args.max_cases is None else min(len(dataset), args.max_cases)
    full_acc, surface_acc, norm_acc = init_accumulator(), init_accumulator(), init_accumulator()
    rows, predictions = [], []

    for i in tqdm(range(n_cases), desc=f"Evaluate {args.split}"):
        graph = dataset[i]
        pred_n, pred, target_n, target = predict_graph(model, graph, coef_norm, device)
        surface = graph.is_airfoil.cpu().numpy() > 0.5
        update_accumulator(full_acc, pred, target)
        update_accumulator(surface_acc, pred, target, surface)
        update_accumulator(norm_acc, pred_n, target_n)

        row = {"index": i, "case_name": dataset.data_names[i], "num_points": int(pred.shape[0]), "surface_points": int(surface.sum())}
        for j, field in enumerate(OUTPUT_FIELDS):
            prefix = field.lower().replace("-", "_")
            for key, value in scalar_metrics(pred[:, j], target[:, j]).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
        if args.save_predictions:
            predictions.append({"case_name": dataset.data_names[i], "prediction": pred, "target": target, "is_airfoil": surface})

    summary = {
        "checkpoint": str(ckpt_path),
        "device": device,
        "split": args.split,
        "num_cases": n_cases,
        "full_field_physical": finalize_accumulator(full_acc),
        "surface_physical": finalize_accumulator(surface_acc),
        "full_field_normalized": finalize_accumulator(norm_acc),
    }
    for section in ["full_field_physical", "surface_physical", "full_field_normalized"]:
        metrics = summary[section]
        metrics["macro_average"] = {
            key: float(np.nanmean([metrics[f][key] for f in OUTPUT_FIELDS]))
            for key in ["mse", "rmse", "mae", "relative_l2", "r2"]
        }

    out_dir = run_dir / "evaluation" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=True)
    with open(out_dir / "metrics_per_case.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index", "case_name"])
        writer.writeheader()
        writer.writerows(rows)
    if args.save_predictions:
        torch.save(predictions, out_dir / "predictions.pt")

    print(f"\nCheckpoint: {ckpt_path}\nCases: {n_cases} ({args.split})\nOutput: {out_dir}")
    print(f"\n{'field':24s} {'RMSE':>12s} {'MAE':>12s} {'rel-L2':>12s} {'R2':>12s}")
    for field in OUTPUT_FIELDS:
        m = summary["full_field_physical"][field]
        print(f"{field:24s} {m['rmse']:12.5e} {m['mae']:12.5e} {m['relative_l2']:12.5e} {m['r2']:12.5f}")


if __name__ == "__main__":
    main()
