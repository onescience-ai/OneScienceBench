from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def kovasznay_solution(xy: np.ndarray | torch.Tensor, nue: float | np.ndarray | torch.Tensor):
    is_torch = isinstance(xy, torch.Tensor)
    xp = torch if is_torch else np
    x = xy[..., 0:1]
    y = xy[..., 1:2]
    if is_torch:
        nue_value = torch.as_tensor(nue, dtype=xy.dtype, device=xy.device)
    else:
        nue_value = nue
    zeta = (0.5 / nue_value) - xp.sqrt((1.0 / (4.0 * nue_value**2)) + 4.0 * math.pi**2)
    u = 1.0 - xp.exp(zeta * x) * xp.cos(2.0 * math.pi * y)
    v = (zeta / (2.0 * math.pi)) * xp.exp(zeta * x) * xp.sin(2.0 * math.pi * y)
    p = 0.5 * (1.0 - xp.exp(2.0 * zeta * x))
    return xp.concatenate([u, v, p], axis=-1) if xp is np else torch.cat([u, v, p], dim=-1)


def latin_hypercube_2d(
    n_samples: int,
    start: Iterable[float],
    stop: Iterable[float],
    seed: int = 1234,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = np.asarray(start, dtype=np.float32)
    stop = np.asarray(stop, dtype=np.float32)
    values = np.empty((int(n_samples), 2), dtype=np.float32)
    for dim in range(2):
        perm = rng.permutation(n_samples)
        values[:, dim] = (perm + rng.random(n_samples)) / n_samples
    return start + values * (stop - start)


def boundary_sequence(
    seq_len: int,
    start: Iterable[float],
    stop: Iterable[float],
    seed: int = 1234,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0, y0 = start
    x1, y1 = stop
    left = np.column_stack([np.full(seq_len, x0), rng.uniform(y0, y1, seq_len)])
    right = np.column_stack([np.full(seq_len, x1), rng.uniform(y0, y1, seq_len)])
    top = np.column_stack([rng.uniform(x0, x1, seq_len), np.full(seq_len, y1)])
    bottom = np.column_stack([rng.uniform(x0, x1, seq_len), np.full(seq_len, y0)])
    return np.concatenate([left, right, top, bottom], axis=0).astype(np.float32)


def make_grid(grid_size: int, start: Iterable[float], stop: Iterable[float]) -> np.ndarray:
    x = np.linspace(start[0], stop[0], int(grid_size), dtype=np.float32)
    y = np.linspace(start[1], stop[1], int(grid_size), dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def build_kovasznay_batch(
    nue_values: Iterable[float],
    domain_samples: int = 32,
    seq_len: int = 5,
    start: tuple[float, float] = (-0.5, -0.5),
    stop: tuple[float, float] = (1.0, 1.5),
    seed: int = 1234,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    query_np = latin_hypercube_2d(domain_samples, start, stop, seed)
    boundary_np = boundary_sequence(seq_len, start, stop, seed + 17)
    queries = []
    targets = []
    boundary_xy = []
    boundary_values = []
    nues = []
    for nue in nue_values:
        queries.append(query_np)
        targets.append(kovasznay_solution(query_np, float(nue)))
        boundary_xy.append(boundary_np)
        boundary_values.append(kovasznay_solution(boundary_np, float(nue)))
        nues.append(float(nue))

    def tensor(values):
        return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)

    return {
        "query_xy": tensor(queries),
        "target": tensor(targets),
        "boundary_xy": tensor(boundary_xy),
        "boundary_values": tensor(boundary_values),
        "nue": tensor(nues).view(-1, 1, 1),
    }
