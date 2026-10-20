# -*- coding: utf-8 -*-
"""MMPINM multi-frequency 2D standard implementation: network definition.

Paper: "A practical PINN framework for multi-scale problems with
multi-magnitude loss terms" (arXiv:2308.06672), Section 4.5 — MMPINN-MFF.

This is the reference model for the codegen eval case `mmpinn_2d_V1`.
The constructor contract (layers/lb/ub/sigma) and the state_dict layout are
locked to the official checkpoint (model_2d.pth), so the official weights load
with strict=True and forward output equals the official reference network.

Architecture (MFF / multi-scale Fourier feature):
    input (x, t) -> map to [-1, 1] -> frozen Fourier feature map (sigma=10)
    -> tanh FC layers [300, 300, 300, 300] -> linear output dim 1
"""
import numpy as np
import torch
import torch.nn as nn


def xavier_std(in_dim, out_dim):
    """Xavier init std (multi-frequency case): sqrt(2/(in+out))."""
    return np.sqrt(2.0 / (in_dim + out_dim))


class PINN2D(nn.Module):
    """Multi-scale Fourier feature (MFF) network for the 2D heat equation.

    state_dict contract (must match official checkpoint exactly):
        lb / ub / W  : registered buffers
        weights.i    : ParameterList entries, shape (in_i, out_i)
        biases.i     : ParameterList entries, shape (1, out_i)
    """

    def __init__(self, layers, lb, ub, sigma=10.0):
        super().__init__()
        self.layers = layers
        # domain mapping buffers (move with .to(device))
        self.register_buffer('lb', torch.tensor(lb, dtype=torch.float32).view(1, 2))
        self.register_buffer('ub', torch.tensor(ub, dtype=torch.float32).view(1, 2))

        # FC weights/biases (order matches official state_dict)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        prev_dim = layers[0]  # Fourier feature dim
        for l in range(len(layers) - 1):
            out_dim = layers[l + 1]
            std = xavier_std(prev_dim, out_dim)
            W = self._truncated_normal(prev_dim, out_dim, std)
            b = torch.randn(1, out_dim)
            self.weights.append(nn.Parameter(W))
            self.biases.append(nn.Parameter(b))
            prev_dim = out_dim

        # frozen Fourier mapping: W ~ N(0, sigma^2), shape (2, layers[0]//2)
        W_fourier = torch.randn(2, layers[0] // 2) * sigma
        self.register_buffer('W', W_fourier)

    @staticmethod
    def _truncated_normal(rows, cols, std):
        # rejection sampling approximating TF truncated_normal(0, std, [-2std, 2std])
        W = torch.zeros(rows, cols)
        filled = torch.zeros(rows, cols, dtype=torch.bool)
        while not filled.all():
            cand = torch.randn(rows, cols) * std
            mask = cand.abs() <= 2 * std
            W = torch.where(mask, cand, W)
            filled = filled | mask
        return W

    def forward(self, X):
        # map input to [-1, 1]
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        # Fourier features
        H = torch.cat([torch.sin(H @ self.W), torch.cos(H @ self.W)], dim=1)
        num_layers = len(self.layers)
        for l in range(num_layers - 2):
            H = torch.tanh(H @ self.weights[l] + self.biases[l])
        return H @ self.weights[-1] + self.biases[-1]
