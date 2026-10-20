"""Configuration loading and paper-contract validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


PAPER_CONTRACT = {
    "experiment.train_simulations": 103,
    "experiment.test_id_simulations": 200,
    "experiment.test_ood_simulations": 496,
    "experiment.repeats": 8,
    "training.epochs": 600,
    "training.batch_size": 1,
    "training.supervised_points": 32000,
    "training.max_lr": 1e-3,
    "model.surface_layers": 4,
    "model.surface_radius": 0.05,
    "model.surface_max_neighbors": 8,
    "model.surf2vol_k": 8,
    "model.n_basis": 8,
}

PAPER_FIELDS = ["x-velocity", "y-velocity", "pressure", "turbulent_viscosity"]


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML and reject accidental deviations from explicit paper settings."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    validate_paper_contract(config)
    return config


def _lookup(config: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"missing required configuration key: {dotted_key}")
        value = value[key]
    return value


def validate_paper_contract(config: Dict[str, Any]) -> None:
    """Enforce only values explicitly fixed by arXiv:2412.09399v2."""
    for key, expected in PAPER_CONTRACT.items():
        actual = _lookup(config, key)
        if actual != expected:
            raise ValueError(f"{key}={actual!r} conflicts with paper value {expected!r}")

    fields = _lookup(config, "experiment.fields")
    if fields != PAPER_FIELDS:
        raise ValueError(f"experiment.fields must preserve paper order: {PAPER_FIELDS}")

    canonical_fields = set(_lookup(config, "model.inlet_canonicalization_fields"))
    expected_fields = {"pressure", "turbulent_viscosity"}
    if canonical_fields != expected_fields:
        raise ValueError(
            "paper Sec. 4.5.1 applies inlet canonicalization to pressure and turbulent_viscosity"
        )
    if not _lookup(config, "model.pressure_signed_log"):
        raise ValueError("paper Sec. 4.5.2 requires signed-log pressure prediction")
