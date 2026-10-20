"""Aerodynamic evaluation metrics for AirfRANS surrogate predictions.

Paper: Hu et al., Physics of Fluids 2025 (DOI 10.1063/5.0258928), Section IV.C.

Design metrics:
    - relative error on lift / drag coefficients: dCL/CL, dCD/CD
    - Spearman rank correlation of force coefficients: rho_L, rho_D
ML metric:
    - per-field MSE on normalized fields (surface pressure p|S, volume fields)

Aerodynamic force coefficients are computed by integrating pressure over the
aerofoil surface, then normalizing by the free-stream dynamic pressure. This
requires *denormalizing* the predicted fields first.
"""

from __future__ import annotations

import numpy as np
import torch

GAMMA = 1.4  # ratio of specific heats for air


def denormalize(
    value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return value * std + mean


def _surface_force_coefficients(
    surf_pred: torch.Tensor,  # [N_surf, 1] denormalized surface pressure
    surf_pos: torch.Tensor,  # [N_surf, 2] surface node positions (x, y)
    u_inf: float,
    rho: float = 1.0,
) -> tuple[float, float]:
    """Integrate surface pressure into lift/drag coefficients."""
    p = surf_pred.detach().cpu().numpy().reshape(-1)
    pts = surf_pos.detach().cpu().numpy()
    n = len(p)
    if n < 3:
        return 0.0, 0.0

    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    poly = pts[order]
    p = p[order]

    edges = np.roll(poly, -1, axis=0) - poly
    mid = (poly + np.roll(poly, -1, axis=0)) / 2.0
    edge_len = np.linalg.norm(edges, axis=1)
    tangent = edges / edge_len[:, None]
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    outward = np.sign((mid - centroid) * normal).sum(axis=1)
    normal[outward < 0] *= -1.0

    force = -np.sum((p[:, None] - p.mean()) * normal * edge_len[:, None], axis=0)

    chord = np.max(pts[:, 0]) - np.min(pts[:, 0])
    if chord <= 0:
        return 0.0, 0.0
    q_inf = 0.5 * rho * u_inf * u_inf
    if q_inf <= 0:
        return 0.0, 0.0

    f_lift = -force[1]
    f_drag = force[0]
    cl = f_lift / (q_inf * chord)
    cd = f_drag / (q_inf * chord)
    return float(cl), float(cd)


def compute_design_metrics(
    surf_pred: torch.Tensor,  # denormalized [N_surf, 1]
    surf_pos: torch.Tensor,  # [N_surf, 2]
    surf_target: torch.Tensor,  # denormalized [N_surf, 1]
    u_inf: float,
) -> dict[str, float]:
    """Relative error of CL/CD for one sample."""
    cl_pred, cd_pred = _surface_force_coefficients(surf_pred, surf_pos, u_inf)
    cl_true, cd_true = _surface_force_coefficients(surf_target, surf_pos, u_inf)

    def rel_err(pred: float, true: float) -> float:
        if abs(true) < 1e-12:
            return 0.0
        return float(abs(pred - true) / abs(true))

    return {
        "relative_error_force_x": rel_err(cl_pred, cl_true),
        "relative_error_force_y": rel_err(cd_pred, cd_true),
    }


def compute_spearman_batch(
    cl_preds: np.ndarray,
    cd_preds: np.ndarray,
    cl_trues: np.ndarray,
    cd_trues: np.ndarray,
) -> dict[str, float]:
    from scipy.stats import spearmanr

    def sr(a, b):
        if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
            return 0.0
        rho, _ = spearmanr(a, b)
        return float(rho) if not np.isnan(rho) else 0.0

    return {
        "spearman_coef_x": sr(cl_preds, cl_trues),
        "spearman_coef_y": sr(cd_preds, cd_trues),
    }


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).detach().cpu())


def compute_normalized_mse_metrics(
    out_surf: torch.Tensor,  # [N_surf, 1] normalized pred
    out_vol: torch.Tensor,  # [N_vol, 4] normalized pred
    target_surf: torch.Tensor,  # [N_surf, 1] normalized target
    target_vol: torch.Tensor,  # [N_vol, 4] normalized target
) -> dict[str, float]:
    """MSE per field on normalized predictions, scaled by 1e-2 like the paper."""
    mse_surf = compute_mse(out_surf, target_surf) * 1e-2
    mse_vol = compute_mse(out_vol, target_vol) * 1e-2
    mse_u = compute_mse(out_vol[:, 0], target_vol[:, 0]) * 1e-2
    mse_v = compute_mse(out_vol[:, 1], target_vol[:, 1]) * 1e-2
    mse_p = compute_mse(out_vol[:, 2], target_vol[:, 2]) * 1e-2
    mse_nut = compute_mse(out_vol[:, 3], target_vol[:, 3]) * 1e-2
    return {
        "surface_p": mse_surf,
        "volume_p": mse_p,
        "volume_u": mse_u,
        "volume_v": mse_v,
        "volume_nut": mse_nut,
        "volume_all": mse_vol,
    }