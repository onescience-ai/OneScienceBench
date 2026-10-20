from __future__ import annotations

import math

import torch


def make_grid(grid_size: int = 16, device: torch.device | None = None) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, int(grid_size), device=device)
    try:
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(coords, coords)
    return torch.stack([xx, yy], dim=-1)


def periodic_vortex_sequence(
    grid_size: int = 16,
    total_len: int = 5,
    batch_size: int = 2,
    viscosity: float = 1.0e-3,
    seed: int = 2026,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a small divergence-free 2D vortex rollout with channels [u, v, omega]."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    grid = make_grid(grid_size, device=device)
    x = grid[..., 0][None, None]
    y = grid[..., 1][None, None]
    times = torch.linspace(0.0, 1.0, int(total_len), device=device)[None, :, None, None]
    phase = torch.rand(int(batch_size), 1, 1, 1, generator=generator, device=device)
    speed_x = 0.15 + 0.05 * torch.rand(int(batch_size), 1, 1, 1, generator=generator, device=device)
    speed_y = 0.10 + 0.05 * torch.rand(int(batch_size), 1, 1, 1, generator=generator, device=device)
    decay = torch.exp(-float(viscosity) * times * (8.0 * math.pi**2))
    shifted_x = x - speed_x * times - phase
    shifted_y = y - speed_y * times + phase
    sx = torch.sin(2.0 * math.pi * shifted_x)
    sy = torch.sin(2.0 * math.pi * shifted_y)
    cx = torch.cos(2.0 * math.pi * shifted_x)
    cy = torch.cos(2.0 * math.pi * shifted_y)
    u = cy * sx * decay
    v = -cx * sy * decay
    omega = 2.0 * sx * sy * decay
    sequence = torch.stack([u, v, omega], dim=-1).to(torch.float32)
    time_values = times[..., 0].expand(int(batch_size), -1, -1)
    return sequence, time_values.to(torch.float32)


def build_ns2d_batch(
    batch_size: int = 2,
    total_len: int = 5,
    grid_size: int = 16,
    viscosity: float = 1.0e-3,
    seed: int = 2026,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    sequence, times = periodic_vortex_sequence(
        grid_size=grid_size,
        total_len=total_len,
        batch_size=batch_size,
        viscosity=viscosity,
        seed=seed,
        device=device,
    )
    return {
        "sequence": sequence,
        "times": times,
        "target": sequence[:, 1:],
        "data_mask": torch.ones(1, 1, 1, 1, sequence.shape[-1], device=sequence.device, dtype=sequence.dtype),
    }
