"""Build and initialize a standalone MatterGen scalar-property adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from portable_mattergen import GemNetTAdapterMixin, MatterGenPropertyAdapter
from portable_mattergen.gemnet import GemNetT
from portable_mattergen.layers.embedding_block import AtomEmbedding


CHECKPOINT_PREFIX = "diffusion_module.model."
ADAPTER_STATE_MARKERS = (
    "gemnet.cond_adapt_layers.",
    "gemnet.cond_mixin_layers.",
    "property_embeddings_adapt.",
)


class GemNetTPropertyAdapter(GemNetTAdapterMixin, GemNetT):
    """GemNet-T with a zero-initialized layerwise property-control path."""

    def __init__(
        self,
        condition_on_adapt: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.initialize_condition_adapter(
            condition_on_adapt=tuple(condition_on_adapt),
            hidden_dim=int(kwargs.get("emb_size_atom", 512)),
            num_blocks=self.num_blocks,
        )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration must be a mapping")
    return config


def build_model(config: dict[str, Any]) -> MatterGenPropertyAdapter:
    model_config = dict(config["model"])
    hidden_dim = int(model_config["hidden_dim"])
    property_name = str(config["finetuning"]["property_name"])
    gemnet_config = dict(model_config["gemnet"])
    atom_embedding = AtomEmbedding(
        emb_size=int(gemnet_config["emb_size_atom"]),
        with_mask_type=True,
    )
    gemnet = GemNetTPropertyAdapter(
        condition_on_adapt=(property_name,),
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
    return MatterGenPropertyAdapter(
        gemnet=gemnet,
        property_name=property_name,
        hidden_dim=hidden_dim,
        denoise_atom_types=bool(model_config["denoise_atom_types"]),
        atom_type_diffusion=str(model_config["atom_type_diffusion"]),
    )


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    return payload


def extract_base_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("checkpoint must contain a non-empty state_dict")
    invalid = [key for key in raw_state if not key.startswith(CHECKPOINT_PREFIX)]
    if invalid:
        raise ValueError(f"unexpected checkpoint state keys: {invalid[:5]}")
    state = {
        key[len(CHECKPOINT_PREFIX) :]: value for key, value in raw_state.items()
    }
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("checkpoint model state must contain tensors only")
    return state


def is_adapter_state(name: str) -> bool:
    return any(marker in name for marker in ADAPTER_STATE_MARKERS)


def is_adapter_parameter(name: str) -> bool:
    return name.startswith("gemnet.cond_adapt_layers.") or name.startswith(
        "gemnet.cond_mixin_layers."
    )


def load_pretrained_model(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> MatterGenPropertyAdapter:
    finetuning = config["finetuning"]
    path = Path(checkpoint_path or finetuning["checkpoint_path"])
    torch.manual_seed(int(finetuning["seed"]))
    model = build_model(config)
    pretrained_state = extract_base_model_state(load_checkpoint_payload(path))
    complete_state = model.state_dict()
    expected_base_keys = {key for key in complete_state if not is_adapter_state(key)}
    if set(pretrained_state) != expected_base_keys:
        missing = sorted(expected_base_keys - set(pretrained_state))
        unexpected = sorted(set(pretrained_state) - expected_base_keys)
        raise RuntimeError(
            "pretrained state does not exactly match the local MatterGen base: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for name, value in pretrained_state.items():
        if tuple(value.shape) != tuple(complete_state[name].shape):
            raise RuntimeError(
                f"checkpoint shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(complete_state[name].shape)}"
            )
        complete_state[name] = value
    model.load_state_dict(complete_state, strict=True)

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_adapter_parameter(name))
    if bool(finetuning.get("full_finetuning", False)):
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return model.to(device)


__all__ = [
    "ADAPTER_STATE_MARKERS",
    "CHECKPOINT_PREFIX",
    "GemNetTPropertyAdapter",
    "MatterGenPropertyAdapter",
    "build_model",
    "extract_base_model_state",
    "is_adapter_parameter",
    "is_adapter_state",
    "load_checkpoint_payload",
    "load_config",
    "load_pretrained_model",
]
