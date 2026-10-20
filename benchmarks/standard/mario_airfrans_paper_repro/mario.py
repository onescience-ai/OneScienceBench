"""MARIO model and training utilities for the standalone AirfRANS task.

This flat module contains the only model-side pieces needed by:
- train_sdf.py: geometry SDF meta-learning
- train.py: conditioned flow-field training
- evaluate.py: checkpoint loading/inference
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def shift_modulation(position, features, layers, activation, with_batch=True):
    """Shift modulation used by MARIO's hypernetwork-conditioned INR."""
    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position
    for i, layer in enumerate(layers):
        res = layer(h)
        h = res * features[..., i, :] + features[..., i, :] + res
        h = activation(h)
    return h


class LatentToModulation(nn.Module):
    def __init__(
        self, latent_dim, num_modulations, dim_hidden, num_layers, activation=nn.SiLU
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation = activation
        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation()]
            if num_layers > 2:
                for _ in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), self.activation()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


class GaussianEncoding(nn.Module):
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale
        else:
            bvals = 2.0 ** torch.linspace(0, scale, embedding_size // 2) - 1
            if dims == 1:
                bvals = bvals[:, None]
            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)
            else:
                tmp = (dims - 1) * (torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)
                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)
        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor):
        return torch.cat(
            [
                self.avals * torch.sin((2.0 * np.pi * tensor) @ self.bvals.T),
                self.avals * torch.cos((2.0 * np.pi * tensor) @ self.bvals.T),
            ],
            dim=-1,
        )


class ModulatedFourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        num_frequencies=8,
        latent_dim=128,
        width=256,
        depth=3,
        depth_hnn=1,
        modulate_scale=False,
        modulate_shift=True,
        frequency_embedding="nerf",
        include_input=True,
        scale=5,
        max_frequencies=32,
        base_frequency=1.25,
    ):
        super().__init__()
        self.frequency_embedding = frequency_embedding
        self.include_input = include_input
        self.scale = scale
        self.embedding = GaussianEncoding(
            embedding_size=num_frequencies * 2, scale=scale, dims=input_dim
        )
        embed_dim = num_frequencies * 2 + (input_dim if include_input else 0)
        self.in_channels = [embed_dim] + [width] * (depth - 1)
        self.out_channels = [width] * (depth - 1) + [output_dim]
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.in_channels[k], self.out_channels[k])
                for k in range(depth)
            ]
        )
        self.depth = depth
        self.hidden_dim = width
        self.depth_hnn = depth_hnn
        self.num_modulations = self.hidden_dim * (self.depth - 1)
        if modulate_scale and modulate_shift:
            self.num_modulations *= 2
        self.latent_to_modulation = LatentToModulation(
            self.latent_dim,
            self.num_modulations,
            dim_hidden=128,
            num_layers=self.depth_hnn,
        )
        self.conditioning = shift_modulation

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        features = self.latent_to_modulation(z)
        position = self.embedding(x)
        if self.frequency_embedding == "gaussian" and self.include_input:
            position = torch.cat([position, x], axis=-1)
        pre_out = self.conditioning(
            position, features, self.layers[:-1], torch.relu
        )
        out = self.layers[-1](pre_out)
        return out.view(*x_shape, out.shape[-1])


class MultiScaleModulatedFourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim=5,
        output_dim=1,
        num_frequencies=8,
        latent_dim=4,
        width=256,
        depth=3,
        depth_hnn=3,
        include_input=True,
        scales=(1, 5),
        conditioning_type="shift_modulation",
        num_heads=4,
        scalar_hidden_dim=128,
        scalar_out_dim=6,
    ):
        super().__init__()
        self.include_input = include_input
        self.scales = list(scales)
        self.conditioning_type = conditioning_type
        self.embeddings = nn.ModuleList(
            [
                GaussianEncoding(
                    embedding_size=num_frequencies * 2,
                    scale=scale,
                    dims=input_dim,
                )
                for scale in self.scales
            ]
        )
        embed_dim = num_frequencies * 2 + (input_dim if include_input else 0)
        self.in_channels = [embed_dim] + [width] * (depth - 1)
        self.out_channels = [width] * (depth - 1) + [width]
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.in_channels[k], self.out_channels[k])
                for k in range(depth)
            ]
        )
        self.final_linear = nn.Linear(len(self.scales) * width, output_dim)
        self.depth = depth
        self.hidden_dim = width
        self.depth_hnn = depth_hnn
        self.num_modulations = self.hidden_dim * (self.depth - 1)
        self.cond_to_modulation = LatentToModulation(
            self.latent_dim,
            self.num_modulations,
            dim_hidden=256,
            num_layers=self.depth_hnn,
        )
        if self.conditioning_type == "shift_modulation":
            self.conditioning = shift_modulation
        else:
            raise ValueError("Only 'shift_modulation' is supported in AirfRANS MARIO.")

        self.scalar_out_dim = scalar_out_dim or 0
        if self.scalar_out_dim > 0:
            self.scalar_head = nn.Sequential(
                nn.Linear(width * (depth - 1), scalar_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(scalar_hidden_dim, self.scalar_out_dim),
            )

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        features = self.cond_to_modulation(z)
        positions = [embedding(x) for embedding in self.embeddings]
        if self.include_input:
            positions = [torch.cat([pos, x], axis=-1) for pos in positions]
        pre_outs = [
            self.conditioning(pos, features, self.layers[:-1], torch.relu)
            for pos in positions
        ]
        outs = [self.layers[-1](pre_out) for pre_out in pre_outs]
        concatenated_out = torch.cat(outs, axis=-1)
        final_out = self.final_linear(concatenated_out)
        return final_out.view(*x_shape, final_out.shape[-1])

    def predict_scalars(self, z):
        if self.scalar_out_dim <= 0:
            raise RuntimeError("scalar_out_dim is 0; no scalar head was created")
        mod_feats = self.cond_to_modulation(z)
        return self.scalar_head(mod_feats)


def create_inr_instance(cfg, input_dim=1, output_dim=1, device="cuda"):
    """Build only the model types used by airfrans_task.

    The upstream file also imported a Tensile2d-task MoE model at module import time.
    That dependency is intentionally removed here so this package is standalone.
    """
    device = torch.device(device)
    if cfg.inr.model_type == "ms_fourier_features":
        return MultiScaleModulatedFourierFeatures(
            input_dim=input_dim,
            output_dim=output_dim,
            num_frequencies=cfg.inr.num_frequencies,
            latent_dim=cfg.inr.latent_dim,
            width=cfg.inr.hidden_dim,
            depth=cfg.inr.depth,
            depth_hnn=cfg.inr.hypernet_depth,
            include_input=cfg.inr.include_input,
            scales=cfg.inr.scale,
            conditioning_type="shift_modulation",
            scalar_out_dim=cfg.inr.out_scalar_dim,
        ).to(device)
    if cfg.inr.model_type == "fourier_features":
        return ModulatedFourierFeatures(
            input_dim=input_dim,
            output_dim=output_dim,
            num_frequencies=cfg.inr.num_frequencies,
            latent_dim=cfg.inr.latent_dim,
            width=cfg.inr.hidden_dim,
            depth=cfg.inr.depth,
            depth_hnn=cfg.inr.hypernet_depth,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            frequency_embedding=cfg.inr.frequency_embedding,
            include_input=cfg.inr.include_input,
            scale=cfg.inr.scale,
            max_frequencies=cfg.inr.max_frequencies,
            base_frequency=cfg.inr.base_frequency,
        ).to(device)
    raise ValueError(f"Unsupported AirfRANS model_type: {cfg.inr.model_type}")


def load_inr(model_weights, cfg, input_dim, output_dim, device="cuda"):
    inr = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device=device
    )
    inr.load_state_dict(model_weights)
    inr.eval()
    return inr


def training_step(
    func_rep,
    graph,
    latent_cond=True,
    predict_scalars=True,
    return_reconstructions=False,
):
    func_rep.zero_grad()
    conditions = graph.cond
    coords = graph.input
    features = graph.output
    if predict_scalars:
        features_scalars = graph.output_scalars

    scalar_loss = torch.tensor(0.0, device=features.device)
    with torch.set_grad_enabled(True):
        if latent_cond:
            features_recon = func_rep.modulated_forward(
                coords, conditions[graph.batch]
            )
            if predict_scalars:
                scalar_pred = func_rep.predict_scalars(conditions)
                scalar_loss = ((scalar_pred - features_scalars) ** 2).mean()
        else:
            features_recon = func_rep(coords)

        mse_components = ((features_recon - features) ** 2).mean(dim=0)
        field_loss = mse_components.mean()
        loss = field_loss + 0.5 * scalar_loss

    outputs = {
        "loss": loss,
        "field_loss": field_loss,
        "scalar_loss": scalar_loss,
        "field_loss_components": mse_components.detach(),
    }
    if return_reconstructions:
        outputs["reconstructions"] = features_recon
    return outputs


def graph_inner_loop(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_steps,
    inner_lr,
    is_train=False,
):
    fitted_modulations = modulations
    for _ in range(inner_steps):
        fitted_modulations = graph_inner_loop_step(
            func_rep,
            fitted_modulations,
            coords,
            features,
            batch_index,
            inner_lr,
            is_train,
        )
    return fitted_modulations


def graph_inner_loop_step(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_lr,
    is_train=False,
):
    batch_size = modulations.shape[0]
    with torch.enable_grad():
        features_recon = func_rep.modulated_forward(
            coords, modulations[batch_index]
        )
        loss = ((features_recon - features) ** 2).mean() * batch_size
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train,
        )[0]
    return modulations - inner_lr * grad


def graph_outer_step(
    func_rep,
    graph,
    inner_steps,
    inner_lr,
    is_train=False,
    detach_modulations=False,
):
    func_rep.zero_grad()
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = torch.zeros_like(graph.modulations).requires_grad_()
    coords = graph.input
    features = graph.output
    modulations = graph_inner_loop(
        func_rep,
        modulations,
        coords,
        features,
        graph.batch,
        inner_steps,
        inner_lr,
        is_train,
    )
    if detach_modulations:
        modulations = modulations.detach()

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(
            coords, modulations[graph.batch]
        )
        loss = ((features_recon - features) ** 2).mean()

    return {"loss": loss, "modulations": modulations}
