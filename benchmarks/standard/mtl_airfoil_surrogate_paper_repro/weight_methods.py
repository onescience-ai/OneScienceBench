"""Multi-task loss weighting methods: STCH, FairGrad, FAMO.

Paper: Hu et al., Physics of Fluids 2025 (DOI 10.1063/5.0258928), Section III.B.

STCH is the paper's main method. FairGrad and FAMO are compared baselines.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class STCH:
    """Smooth Tchebycheff Scalarization.

    g_mu(x|lambda) = mu * log( sum_i exp( lambda_i * (f_i(x) - z_i*) / mu ) )

    Details (official repo):
    - mu = 1
    - first ``warmup_epoch`` calls use sum_i log(L_i + eps)
    - the first non-warmup call records the current mean loss as the nadir
    - preference vector lambda is uniform (lambda_i = 1 / n_tasks)
    """

    def __init__(
        self,
        mu: float = 1.0,
        warmup_epoch: int = 4,
        n_tasks: int = 2,
        eps: float = 1e-20,
    ):
        self.mu = mu
        self.warmup_epoch = warmup_epoch
        self.n_tasks = n_tasks
        self.eps = eps
        self.nadir: torch.Tensor | None = None
        self.step = 0

    def backward(
        self,
        losses: Sequence[torch.Tensor],
        shared_parameters: Sequence[torch.nn.Parameter] | None = None,
        task_specific_parameters: Sequence[torch.nn.Parameter] | None = None,
        last_shared_parameters: Sequence[torch.nn.Parameter] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        losses = torch.stack(list(losses))

        if self.step < self.warmup_epoch:
            loss = torch.sum(torch.log(losses + self.eps))
        else:
            if self.nadir is None:
                self.nadir = torch.mean(losses).detach()
            log_ratios = torch.log(losses / self.nadir)
            loss = self.mu * torch.log(
                torch.sum(torch.exp(log_ratios / self.mu))
            )
            loss = loss * self.n_tasks

        loss.backward()
        self.step += 1
        return loss

    def state_dict(self) -> dict:
        state = {"step": self.step}
        if self.nadir is not None:
            state["nadir"] = self.nadir.detach().cpu().numpy().tolist()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.step = state.get("step", 0)
        if "nadir" in state and state["nadir"] is not None:
            self.nadir = torch.as_tensor(state["nadir"])


class FairGrad:
    """FairGrad gradient manipulation (compared baseline).

    Solves G^T G gamma = gamma^(-1/alpha) (alpha = 1) with
    scipy.optimize.least_squares, then applies the task-weighted shared
    gradient. Requires an optimizer reference for manual parameter update.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.step = 0

    def backward(
        self,
        losses: Sequence[torch.Tensor],
        shared_parameters: Sequence[torch.nn.Parameter] | None = None,
        task_specific_parameters: Sequence[torch.nn.Parameter] | None = None,
        last_shared_parameters: Sequence[torch.nn.Parameter] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from scipy.optimize import least_squares

        optimizer = kwargs.get("optimizer")
        if shared_parameters is None or optimizer is None:
            raise ValueError(
                "FairGrad requires shared_parameters and an optimizer "
                "(manual gradient update)."
            )
        shared = list(shared_parameters)
        lr = optimizer.param_groups[0]["lr"]

        def flatten(grads):
            return torch.cat([g.reshape(-1) for g in grads])

        per_task_shared = []
        for loss in losses:
            grads = torch.autograd.grad(
                loss, shared, retain_graph=True, allow_unused=True
            )
            grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, shared)
            ]
            per_task_shared.append(grads)

        G = torch.stack([flatten(gs) for gs in per_task_shared])  # [T, d]
        GGt = G @ G.t()

        def residual(gamma):
            return GGt.detach().numpy() @ gamma - np.power(gamma, -1.0 / self.alpha)

        sol = least_squares(residual, np.ones(len(losses)))
        gamma = np.clip(sol.x, 1e-8, None)

        with torch.no_grad():
            for p in shared:
                p.grad = None
            for i, gs in enumerate(per_task_shared):
                for p, g in zip(shared, gs):
                    p.grad = g * gamma[i] if p.grad is None else p.grad + g * gamma[i]
            for p in shared:
                p.data.add_(-lr, p.grad)

        with torch.no_grad():
            for loss, task_params in zip(losses, task_specific_parameters):
                grads = torch.autograd.grad(loss, list(task_params), allow_unused=True)
                for p, g in zip(task_params, grads):
                    if g is not None:
                        p.data.add_(-lr, g)

        self.step += 1
        return torch.stack(list(losses)).sum()


class FAMO:
    """FAMO: fast adaptive multitask optimization (compared baseline)."""

    def __init__(self, lr_z: float = 0.01, temperature: float = 1.0, n_tasks: int = 2):
        self.lr_z = lr_z
        self.temperature = temperature
        self.z = torch.ones(n_tasks)
        self.prev_loss: torch.Tensor | None = None
        self.step = 0

    def backward(
        self,
        losses: Sequence[torch.Tensor],
        shared_parameters: Sequence[torch.nn.Parameter] | None = None,
        task_specific_parameters: Sequence[torch.nn.Parameter] | None = None,
        last_shared_parameters: Sequence[torch.nn.Parameter] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        loss_t = torch.stack(list(losses)).detach()
        if self.prev_loss is not None:
            delta = (loss_t - self.prev_loss) / (self.prev_loss + 1e-12)
            denom = torch.sum(delta * delta)
            if denom > 1e-12:
                self.z = self.z - self.lr_z * delta / denom
        self.prev_loss = loss_t

        weight = torch.softmax(self.z / self.temperature, dim=0)
        loss = torch.sum(weight * torch.stack(list(losses)))
        loss.backward()
        self.step += 1
        return loss

    def state_dict(self) -> dict:
        return {"step": self.step, "z": self.z.detach().cpu().numpy().tolist()}

    def load_state_dict(self, state: dict) -> None:
        self.step = state.get("step", 0)
        if "z" in state:
            self.z = torch.as_tensor(state["z"]).float()


WEIGHT_METHODS = {"stch": STCH, "fairgrad": FairGrad, "famo": FAMO}


def make_weight_method(method: str, **kwargs) -> object:
    """Factory: build a weight method by name (defaults for STCH)."""
    name = method.lower()
    if name not in WEIGHT_METHODS:
        raise ValueError(
            f"Unknown weight_method '{method}'; expected {sorted(WEIGHT_METHODS)}"
        )
    if name == "stch":
        kwargs.setdefault("mu", 1.0)
        kwargs.setdefault("warmup_epoch", 4)
        kwargs.setdefault("n_tasks", 2)
    return WEIGHT_METHODS[name](**kwargs)