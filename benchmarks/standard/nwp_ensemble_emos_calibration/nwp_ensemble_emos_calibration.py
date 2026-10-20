"""Permutation-invariant EMOS-style calibration for NWP ensembles."""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
import numpy as np


class NWPEnsembleEMOS:
    def __init__(self, bias: float = 0.0, scale: float = 1.0, quantile_levels=(0.1, 0.5, 0.9)):
        levels = np.asarray(quantile_levels, dtype=np.float64)
        if levels.ndim != 1 or len(levels) == 0 or np.any(levels <= 0) or np.any(levels >= 1) or np.any(np.diff(levels) <= 0):
            raise ValueError("quantile_levels must be strictly increasing in (0,1)")
        if not np.isfinite(bias) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("bias/scale are invalid")
        self.bias, self.scale, self.quantile_levels = float(bias), float(scale), levels

    def predict(self, ensemble: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(ensemble, dtype=np.float64)
        if values.ndim < 2 or values.shape[1] < 2 or not np.isfinite(values).all():
            raise ValueError("ensemble must be finite with shape [B,M,...], M>=2")
        mean = values.mean(axis=1) + self.bias
        spread = np.maximum(values.std(axis=1) * self.scale, 1e-8)
        quantiles = np.stack(
            [mean + spread * NormalDist().inv_cdf(float(q)) for q in self.quantile_levels],
            axis=-1,
        )
        return {"mean": mean, "spread": spread, "quantiles": quantiles}

    def load_checkpoint(self, path: str | Path) -> "NWPEnsembleEMOS":
        payload = np.load(path, allow_pickle=False)
        if set(payload.files) != {"bias", "scale"}:
            raise ValueError("checkpoint keys must be exactly bias and scale")
        return NWPEnsembleEMOS(float(payload["bias"]), float(payload["scale"]), self.quantile_levels)
