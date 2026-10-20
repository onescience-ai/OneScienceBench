"""Build and strictly load a standalone MatterGen base denoiser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from portable_mattergen import GemNetTDenoiser
from portable_mattergen.gemnet import GemNetT
from portable_mattergen.layers.embedding_block import AtomEmbedding


CHECKPOINT_PREFIX = "diffusion_module.model."


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration must be a mapping")
    return config


def build_model(config: dict[str, Any]) -> GemNetTDenoiser:
    model_config = dict(config["model"])
    hidden_dim = int(model_config["hidden_dim"])
    gemnet_config = dict(model_config["gemnet"])
    atom_embedding = AtomEmbedding(
        emb_size=int(gemnet_config["emb_size_atom"]),
        with_mask_type=True,
    )
    gemnet = GemNetT(
        num_targets=int(gemnet_config["num_targets"]),
        latent_dim=hidden_dim,
        atom_embedding=atom_embedding,
        cutoff=float(gemnet_config["cutoff"]),
        emb_size_atom=int(gemnet_config["emb_size_atom"]),
        emb_size_edge=int(gemnet_config["emb_size_edge"]),
        max_neighbors=int(gemnet_config["max_neighbors"]),
        max_cell_images_per_dim=int(gemnet_config["max_cell_images_per_dim"]),
        num_blocks=int(gemnet_config["num_blocks"]),
        regress_stress=bool(gemnet_config["regress_stress"]),
        otf_graph=bool(gemnet_config["otf_graph"]),
    )
    return GemNetTDenoiser(
        gemnet=gemnet,
        hidden_dim=hidden_dim,
        denoise_atom_types=bool(model_config["denoise_atom_types"]),
        atom_type_diffusion=str(model_config["atom_type_diffusion"]),
    )


def extract_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("checkpoint must contain a non-empty state_dict")
    unexpected = [key for key in raw_state if not key.startswith(CHECKPOINT_PREFIX)]
    if unexpected:
        raise ValueError(f"unexpected checkpoint state keys: {unexpected[:5]}")
    state = {
        key[len(CHECKPOINT_PREFIX) :]: value
        for key, value in raw_state.items()
    }
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("model state must contain tensors only")
    return state


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    return payload


def load_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> GemNetTDenoiser:
    model = build_model(config)
    state = extract_model_state(load_checkpoint_payload(checkpoint_path))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


__all__ = [
    "CHECKPOINT_PREFIX",
    "GemNetTDenoiser",
    "build_model",
    "extract_model_state",
    "load_checkpoint_payload",
    "load_config",
    "load_model",
]
