"""Small neural EnKF-style analysis update with explicit observation masking."""
from __future__ import annotations

from pathlib import Path
import numpy as np


class NeuralEnKFAnalyzer:
    def __init__(self, state_dim: int = 4, obs_dim: int = 2):
        if state_dim <= 0 or obs_dim <= 0 or obs_dim > state_dim:
            raise ValueError("invalid state/observation dimensions")
        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.gain = np.zeros((state_dim, obs_dim), dtype=np.float64)
        self.gain[:obs_dim, :obs_dim] = np.eye(obs_dim) * 0.5
        self.observation_matrix = np.zeros((obs_dim, state_dim), dtype=np.float64)
        self.observation_matrix[:, :obs_dim] = np.eye(obs_dim)

    def analyze(self, prior: np.ndarray, observations: np.ndarray, observed_mask: np.ndarray | None = None) -> np.ndarray:
        prior = np.asarray(prior, dtype=np.float64)
        observations = np.asarray(observations, dtype=np.float64)
        if prior.ndim != 3 or prior.shape[-1] != self.state_dim or observations.shape != (prior.shape[0], self.obs_dim):
            raise ValueError("prior/observations have invalid shapes")
        if not np.isfinite(prior).all() or not np.isfinite(observations).all():
            raise ValueError("prior and observations must be finite")
        if observed_mask is None:
            observed_mask = np.ones_like(observations, dtype=bool)
        observed_mask = np.asarray(observed_mask)
        if observed_mask.shape != observations.shape or observed_mask.dtype != bool:
            raise ValueError("observed_mask must be boolean and match observations")
        predicted = prior @ self.observation_matrix.T
        innovation = (observations - predicted.mean(axis=1)) * observed_mask
        return prior + (innovation @ self.gain.T)[:, None, :]

    def load_checkpoint(self, path: str | Path) -> "NeuralEnKFAnalyzer":
        payload = np.load(path, allow_pickle=False)
        if set(payload.files) != {"gain", "observation_matrix"}:
            raise ValueError("checkpoint keys must be exactly gain and observation_matrix")
        if payload["gain"].shape != self.gain.shape or payload["observation_matrix"].shape != self.observation_matrix.shape:
            raise ValueError("checkpoint shapes do not match")
        result = NeuralEnKFAnalyzer(self.state_dim, self.obs_dim)
        result.gain = np.asarray(payload["gain"], dtype=np.float64)
        result.observation_matrix = np.asarray(payload["observation_matrix"], dtype=np.float64)
        return result
