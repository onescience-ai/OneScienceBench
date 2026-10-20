from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class SDFDataset:
    """Dataset used by the SDF/meta-learning stage."""

    def __init__(
        self,
        data_list,
        data_names,
        is_train=True,
        coef_norm=None,
        num_points=None,
    ):
        self.data_list = data_list
        self.data_names = data_names
        self.num_points = num_points
        self.is_train = is_train
        if is_train:
            self.compute_norm_params(data_list)
        else:
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train=False")
            self.coef_norm = coef_norm
            self.pos_norm = coef_norm["pos_norm"]
        self.processed_dataset = self.process_data(data_list)

    def compute_norm_params(self, data_list):
        self.coef_norm = {}
        all_targets = np.concatenate([data[:, 4:5] for data in data_list])
        self.coef_norm["mean"] = all_targets.mean(axis=0)
        std = all_targets.std(axis=0)
        std[std == 0] = 1.0
        self.coef_norm["std"] = std
        all_positions = np.concatenate([data[:, :2] for data in data_list])
        self.pos_norm = {
            "min": all_positions.min(axis=0),
            "max": all_positions.max(axis=0),
        }
        self.coef_norm["pos_norm"] = self.pos_norm
        return self.coef_norm

    def process_data(self, dataset_list):
        dataset = []
        target_idx = 4
        denom = self.pos_norm["max"] - self.pos_norm["min"]
        denom[denom == 0] = 1.0
        for sim_data in dataset_list:
            pos = sim_data[:, :2]
            pos = 2 * (pos - self.pos_norm["min"]) / denom - 1
            output = sim_data[:, target_idx : target_idx + 1]
            output = (output - self.coef_norm["mean"]) / self.coef_norm["std"]
            dataset.append(
                Data(
                    pos=torch.tensor(pos, dtype=torch.float),
                    input=torch.tensor(pos, dtype=torch.float),
                    output=torch.tensor(output, dtype=torch.float),
                    is_airfoil=torch.tensor(sim_data[:, -1]),
                )
            )
        if self.num_points is not None:
            dataset = subsample_dataset(dataset, self.num_points)
        return dataset


def make_bl_layer(
    sample,
    thickness: float = 0.03,
    decay: Literal["linear", "exponential", "squared"] = "squared",
):
    if not (0 < thickness <= 1):
        raise ValueError("Thickness should be in (0, 1].")

    def threshold(var, value):
        mask = var > value
        var = var.copy()
        var[~mask] = 0
        if mask.any():
            vmax = var[mask].max()
            if vmax > value:
                var[mask] = (var[mask] - value) / (vmax - value)
        return var

    sdf = sample["point_cloud_field/distance_function"]
    df = sdf.max() - sdf
    max_df = df.max()
    if max_df > 0:
        df = df / max_df
    if decay == "squared":
        return threshold(df, 1 - thickness) ** 2
    if decay == "linear":
        return threshold(df, 1 - thickness)
    if decay == "exponential":
        return (np.exp(threshold(df, 1 - thickness)) - 1) / (np.exp(1) - 1)
    raise ValueError(f"Unknown decay type: {decay}")


class AirfransFlowDataset(Dataset):
    """AirfRANS flow dataset used by MARIO.

    Point input: [x, y, sdf, nx, ny, boundary_layer_mask]
    Condition: [8-D geometry latent, inlet_u, inlet_v]
    Target: [u, v, p/rho, nu_t]
    """

    def __init__(
        self,
        data_list,
        data_names,
        geom_latents,
        mode="train",
        coef_norm=None,
        num_points=None,
        idx_list=None,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(None, transform, pre_transform)
        raw_data = list(data_list)
        raw_names = list(data_names)
        raw_latents = np.asarray(geom_latents)

        outlier_names = [
            "airFoil2D_SST_50.077_-4.416_2.834_4.029_1.0_5.156"
        ]
        to_remove = [i for i, n in enumerate(raw_names) if n in outlier_names]
        if to_remove:
            raw_data = [d for i, d in enumerate(raw_data) if i not in to_remove]
            raw_names = [n for i, n in enumerate(raw_names) if i not in to_remove]
            raw_latents = np.delete(raw_latents, to_remove, axis=0)

        if len(raw_data) != len(raw_latents):
            raise ValueError(
                f"data/latent length mismatch: {len(raw_data)} vs {len(raw_latents)}"
            )

        self.raw_data_list = raw_data
        self.data_names = raw_names
        self._raw_geom_latents = raw_latents
        total = len(raw_data)
        self.idx_list = list(range(total)) if idx_list is None else list(idx_list)
        self.geom_latents = raw_latents[self.idx_list]
        self.mode = mode
        self.num_points = num_points

        self.pos_indices = [0, 1]
        self.inlet_vel_indices = [2, 3]
        self.sdf_index = 4
        self.normal_indices = [5, 6]
        self.output_indices = [7, 8, 9, 10]
        self.airfoil_index = 11
        self.bl_thickness = 0.03
        self.bl_decay = "squared"
        self.output_fields = [
            "Velocity-x",
            "Velocity-y",
            "Pressure",
            "Turbulent-viscosity",
        ]

        if mode == "train":
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm
        self.data_list = self.process_dataset()

    def compute_norm_params(self):
        pos_all, sdf_all, normal_all, bl_all = [], [], [], []
        output_fields_all = {i: [] for i in range(len(self.output_indices))}
        cond_all = []
        for local_i, idx in enumerate(self.idx_list):
            data = self.raw_data_list[idx]
            pos = data[:, self.pos_indices]
            sdf = data[:, self.sdf_index]
            normals = data[:, self.normal_indices]
            outputs = data[:, self.output_indices]
            inlet_vel = data[:, self.inlet_vel_indices]
            bl_mask = make_bl_layer(
                {"point_cloud_field/distance_function": sdf},
                thickness=self.bl_thickness,
                decay=self.bl_decay,
            )
            pos_all.append(pos)
            sdf_all.append(sdf)
            normal_all.append(normals)
            bl_all.append(bl_mask)
            for i in range(len(self.output_indices)):
                output_fields_all[i].append(outputs[:, i])
            cond_all.append(
                np.concatenate([self.geom_latents[local_i], np.mean(inlet_vel, axis=0)])
            )

        pos_all = np.vstack(pos_all)
        pos_min, pos_max = pos_all.min(axis=0), pos_all.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0

        sdf_all = np.hstack(sdf_all)
        sdf_min, sdf_max = sdf_all.min(), sdf_all.max()
        sdf_range = sdf_max - sdf_min or 1.0

        normal_all = np.vstack(normal_all)
        normal_min, normal_max = normal_all.min(axis=0), normal_all.max(axis=0)
        normal_range = normal_max - normal_min
        normal_range[normal_range == 0] = 1.0

        bl_all = np.hstack(bl_all)
        bl_min, bl_max = bl_all.min(), bl_all.max()
        bl_range = bl_max - bl_min or 1.0

        output_means, output_stds = {}, {}
        for i, field_name in enumerate(self.output_fields):
            field_data = np.hstack(output_fields_all[i])
            output_means[field_name] = float(field_data.mean())
            output_stds[field_name] = float(field_data.std()) or 1.0

        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std = cond_all.std(axis=0)
        cond_std[cond_std == 0] = 1.0

        return {
            "pos": {"min": pos_min, "max": pos_max, "range": pos_range},
            "sdf": {"min": sdf_min, "max": sdf_max, "range": sdf_range},
            "normal": {
                "min": normal_min,
                "max": normal_max,
                "range": normal_range,
            },
            "bl": {"min": bl_min, "max": bl_max, "range": bl_range},
            "output": {"mean": output_means, "std": output_stds},
            "cond": {"mean": cond_mean, "std": cond_std},
        }

    def process_dataset(self):
        processed = []
        c = self.coef_norm
        for local_i, idx in enumerate(self.idx_list):
            data = self.raw_data_list[idx]
            pos = data[:, self.pos_indices]
            sdf = data[:, self.sdf_index]
            normals = data[:, self.normal_indices]
            airfoil_bool = data[:, self.airfoil_index]
            if self.num_points and pos.shape[0] > self.num_points:
                sel = np.random.choice(pos.shape[0], self.num_points, replace=False)
                pos, sdf, normals, airfoil_bool = (
                    pos[sel],
                    sdf[sel],
                    normals[sel],
                    airfoil_bool[sel],
                )
                data = data[sel]

            bl_mask = make_bl_layer(
                {"point_cloud_field/distance_function": sdf},
                thickness=self.bl_thickness,
                decay=self.bl_decay,
            )
            pos_n = 2 * (pos - c["pos"]["min"]) / c["pos"]["range"] - 1
            sdf_n = 2 * ((sdf - c["sdf"]["min"]) / c["sdf"]["range"]) - 1
            normals_n = (
                2 * (normals - c["normal"]["min"]) / c["normal"]["range"] - 1
            )
            bl_n = 2 * ((bl_mask - c["bl"]["min"]) / c["bl"]["range"]) - 1
            input_feats = np.concatenate(
                [pos_n, sdf_n[:, None], normals_n, bl_n[:, None]], axis=1
            )
            output_data = []
            for j, field_name in enumerate(self.output_fields):
                field_data = data[:, self.output_indices[j]]
                field_norm = (
                    field_data - c["output"]["mean"][field_name]
                ) / c["output"]["std"][field_name]
                output_data.append(field_norm[:, None])

            inlet_vel = np.mean(data[:, self.inlet_vel_indices], axis=0)
            cond = np.concatenate([self.geom_latents[local_i], inlet_vel])
            cond_n = (cond - c["cond"]["mean"]) / c["cond"]["std"]
            processed.append(
                Data(
                    input=torch.tensor(input_feats, dtype=torch.float),
                    pos=torch.tensor(pos_n, dtype=torch.float),
                    is_airfoil=torch.tensor(airfoil_bool, dtype=torch.float),
                    output=torch.tensor(np.hstack(output_data), dtype=torch.float),
                    cond=torch.tensor(cond_n, dtype=torch.float).unsqueeze(0),
                )
            )
        return processed

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def subsample_dataset(dataset, num_points, seed=None):
    if num_points is None:
        return list(dataset)
    rng = np.random.default_rng(seed)
    out = []
    for data in dataset:
        n = data.input.size(0)
        if n <= num_points:
            out.append(data)
            continue
        idx = torch.from_numpy(rng.choice(n, num_points, replace=False)).long()
        kwargs = {
            "input": data.input[idx],
            "pos": data.pos[idx],
        }
        if hasattr(data, "output"):
            kwargs["output"] = data.output[idx]
        if hasattr(data, "is_airfoil"):
            kwargs["is_airfoil"] = data.is_airfoil[idx]
        new_data = Data(**kwargs)
        if hasattr(data, "cond"):
            new_data.cond = data.cond
        if hasattr(data, "output_scalars"):
            new_data.output_scalars = data.output_scalars
        out.append(new_data)
    return out
