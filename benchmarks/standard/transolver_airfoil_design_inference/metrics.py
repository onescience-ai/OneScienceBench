"""Transolver-Airfoil 推理所需的本地评估工具。

本文件只依赖 PyTorch、PyTorch Geometric、NumPy、SciPy 和 PyVista。
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    """Convert NumPy values recursively into JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


class AirfRANSEvaluator:
    """封装 AirfRANS 推理、气动力计算和论文指标汇总。"""

    KINEMATIC_VISCOSITY = np.asarray(1.56e-5)

    def __init__(
        self,
        device: torch.device | str,
        models: Sequence[nn.Module],
        hparams: Sequence[Mapping[str, Any]],
        coef_norm: Sequence[np.ndarray],
    ) -> None:
        if not models:
            raise ValueError("At least one model is required")
        if len(models) != len(hparams):
            raise ValueError("Each model must have one hparams mapping")
        self.device = torch.device(device)
        self.models = list(models)
        self.hparams = list(hparams)
        self.coef_norm = tuple(np.asarray(value) for value in coef_norm)
        if tuple(value.shape for value in self.coef_norm) != ((7,), (7,), (4,), (4,)):
            raise ValueError("coef_norm shapes must be (7,), (7,), (4,), (4,)")

    @staticmethod
    def relative_error(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return np.abs((reference - prediction) / reference)

    @classmethod
    def wall_shear_stress(cls, velocity_jacobian: np.ndarray, normals: np.ndarray) -> np.ndarray:
        strain = 0.5 * (velocity_jacobian + velocity_jacobian.transpose(0, 2, 1))
        strain = strain - strain.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
        stress = 2 * cls.KINEMATIC_VISCOSITY.reshape(-1, 1, 1) * strain
        return (stress * normals[:, :2].reshape(-1, 1, 2)).sum(axis=2)

    @torch.no_grad()
    def infer(self, data: Any) -> tuple[list[torch.Tensor], np.ndarray]:
        """随机分块覆盖完整非结构网格，并对重复点预测取平均。"""
        data = data.to("cpu")
        outputs = [torch.zeros_like(data.y) for _ in self.models]
        prediction_count = torch.zeros_like(data.y[:, :1])
        remaining_points = set(map(tuple, data.pos[:, :2].numpy()))
        iterations = 0
        elapsed_total = np.zeros(len(self.models), dtype=np.float64)

        while remaining_points:
            iterations += 1
            sampled = data.clone()
            count = min(int(self.hparams[0]["subsampling"]), sampled.x.size(0))
            indices = torch.tensor(random.sample(range(sampled.x.size(0)), count))
            remaining_points -= set(map(tuple, sampled.pos[indices, :2].numpy()))
            for name in ("pos", "x", "y", "surf", "batch"):
                if hasattr(sampled, name) and getattr(sampled, name) is not None:
                    setattr(sampled, name, getattr(sampled, name)[indices])

            for model_index, model in enumerate(self.models):
                params = self.hparams[model_index]
                if "r" in params and "max_neighbors" in params:
                    sampled.edge_index = pyg_nn.radius_graph(
                        x=sampled.pos.to(self.device),
                        r=float(params["r"]),
                        loop=True,
                        max_num_neighbors=int(params["max_neighbors"]),
                    )
                else:
                    sampled.edge_index = None
                model.eval()
                sampled_device = sampled.to(self.device)
                start = time.time()
                prediction = model(sampled_device)
                elapsed_total[model_index] += time.time() - start
                partial = torch.zeros_like(data.y)
                partial[indices] = prediction.cpu()
                outputs[model_index] += partial
            prediction_count[indices] += 1

        mean_out, std_out = self.coef_norm[2], self.coef_norm[3]
        for model_index, output in enumerate(outputs):
            output = output / prediction_count
            output[data.surf, :2] = -torch.as_tensor(mean_out[:2]) / (
                torch.as_tensor(std_out[:2]) + 1.0e-8
            )
            output[data.surf, 3] = -torch.as_tensor(mean_out[3]) / (
                torch.as_tensor(std_out[3]) + 1.0e-8
            )
            outputs[model_index] = output
        return outputs, elapsed_total / iterations

    def prediction_meshes(
        self,
        internal: Any,
        aerofoil: Any,
        outputs: Sequence[torch.Tensor],
        surface_mask: torch.Tensor,
    ) -> tuple[list[Any], list[Any]]:
        """反归一化网络输出并写回预测的 VTU/VTP 网格。"""
        from AirfRANSDataSet import reorganize

        mean_out, std_out = self.coef_norm[2], self.coef_norm[3]
        mask = surface_mask.cpu().numpy().astype(bool)
        internals, aerofoils = [], []
        for output in outputs:
            predicted_internal = internal.copy()
            predicted_aerofoil = aerofoil.copy()
            physical = (output * (std_out + 1.0e-8) + mean_out).numpy()
            physical[mask, :2] = 0
            physical[mask, 3] = 0
            predicted_internal.point_data["U"][:, :2] = physical[:, :2]
            predicted_internal.point_data["p"] = physical[:, 2]
            predicted_internal.point_data["nut"] = physical[:, 3]
            predicted_aerofoil.point_data["p"] = reorganize(
                predicted_internal.points[mask, :2],
                predicted_aerofoil.points[:, :2],
                predicted_internal.point_data["p"][mask],
            )
            internals.append(predicted_internal.ptc(pass_point_data=True))
            aerofoils.append(predicted_aerofoil.ptc(pass_point_data=True))
        return internals, aerofoils

    @classmethod
    def compute_coefficients(
        cls,
        internals: Sequence[Any],
        aerofoils: Sequence[Any],
        surface_mask: torch.Tensor,
        u_inf: float,
        angle: float,
        keep_vtk: bool = False,
    ) -> Any:
        """从压力和壁面剪切应力积分得到阻力、升力系数。"""
        from AirfRANSDataSet import reorganize

        mask = surface_mask.cpu().numpy().astype(bool)
        coefficients, output_internals, output_aerofoils = [], [], []
        for internal, aerofoil in zip(internals, aerofoils):
            current_internal = internal.copy()
            current_aerofoil = aerofoil.copy()
            current_internal = current_internal.compute_derivative(
                scalars="U", gradient="pred_grad"
            )
            gradient = current_internal.point_data["pred_grad"].reshape(-1, 3, 3)[
                mask, :2, :2
            ]
            surface_pressure = current_internal.point_data["p"][mask]
            source_points = current_internal.points[mask, :2]
            target_points = current_aerofoil.points[:, :2]
            gradient = reorganize(source_points, target_points, gradient)
            surface_pressure = reorganize(source_points, target_points, surface_pressure)
            current_aerofoil.point_data["wallShearStress"] = cls.wall_shear_stress(
                gradient, -current_aerofoil.point_data["Normals"]
            )
            current_aerofoil.point_data["p"] = surface_pressure
            current_internal = current_internal.ptc(pass_point_data=True)
            current_aerofoil = current_aerofoil.ptc(pass_point_data=True)

            pressure_force = -current_aerofoil.cell_data["p"][:, None] * current_aerofoil.cell_data[
                "Normals"
            ][:, :2]
            shear_force = (
                current_aerofoil.cell_data["wallShearStress"]
                * current_aerofoil.cell_data["Length"].reshape(-1, 1)
            ).sum(axis=0)
            pressure_force = (
                pressure_force * current_aerofoil.cell_data["Length"].reshape(-1, 1)
            ).sum(axis=0)
            force = shear_force - pressure_force
            alpha = angle * np.pi / 180
            rotation = np.asarray(
                [[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]]
            )
            coefficients.append(2 * (rotation @ force) / u_inf**2)
            if keep_vtk:
                output_internals.append(current_internal)
                output_aerofoils.append(current_aerofoil)
        if keep_vtk:
            return coefficients, output_internals, output_aerofoils
        return coefficients

    def evaluate(
        self,
        loader: Iterable[Any],
        sample_names: Sequence[str],
        data_path: str | Path,
        *,
        max_samples: Optional[int] = None,
        save_indices: Sequence[int] = (),
        output_dir: str | Path | None = None,
        model_name: str = "Transolver",
        progress: bool = False,
    ) -> dict[str, Any]:
        """运行数据集推理并汇总体积、表面、气动力和相关性指标。"""
        import pyvista as pv
        import scipy.stats as scipy_stats

        iterator = loader
        if progress:
            from tqdm import tqdm

            iterator = tqdm(loader, desc="Testing")
        count = len(sample_names) if max_samples is None else min(max_samples, len(sample_names))
        selected_to_save = set(save_indices)
        destination = Path(output_dir) if output_dir is not None else None
        if selected_to_save and destination is None:
            raise ValueError("output_dir is required when save_indices is not empty")
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)

        criterion = nn.MSELoss(reduction="none")
        scores_volume, scores_surface, scores_force = [], [], []
        scores_pressure, scores_wss, elapsed_times = [], [], []
        true_coefficients, predicted_coefficients = [], []
        data_root = Path(data_path)

        for index, data in enumerate(iterator):
            if index >= count:
                break
            simulation = sample_names[index]
            u_inf = float(simulation.split("_")[2])
            angle = float(simulation.split("_")[3])
            outputs, elapsed = self.infer(data)
            elapsed_times.append(elapsed)
            internal = pv.read(data_root / simulation / f"{simulation}_internal.vtu")
            aerofoil = pv.read(data_root / simulation / f"{simulation}_aerofoil.vtp")
            true_list, _, true_airfoils = self.compute_coefficients(
                [internal], [aerofoil], data.surf, u_inf, angle, keep_vtk=True
            )
            predicted_internals, predicted_airfoils = self.prediction_meshes(
                internal, aerofoil, outputs, data.surf
            )
            predicted_list, predicted_internals, predicted_airfoils = self.compute_coefficients(
                predicted_internals,
                predicted_airfoils,
                data.surf,
                u_inf,
                angle,
                keep_vtk=True,
            )
            true_coef = np.asarray(true_list[0])
            predicted_coef = np.asarray(predicted_list[0])
            true_coefficients.append(true_coef)
            predicted_coefficients.append(predicted_coef)

            output = outputs[0]
            scores_volume.append(
                criterion(output[~data.surf], data.y[~data.surf]).mean(dim=0).cpu().numpy()
            )
            scores_surface.append(
                criterion(output[data.surf], data.y[data.surf]).mean(dim=0).cpu().numpy()
            )
            scores_force.append(self.relative_error(true_coef, predicted_coef))
            scores_wss.append(
                self.relative_error(
                    true_airfoils[0].point_data["wallShearStress"],
                    predicted_airfoils[0].point_data["wallShearStress"],
                ).mean(axis=0)
            )
            scores_pressure.append(
                self.relative_error(
                    true_airfoils[0].point_data["p"],
                    predicted_airfoils[0].point_data["p"],
                ).mean(axis=0)
            )
            if index in selected_to_save and destination is not None:
                predicted_internals[0].save(destination / f"{simulation}_pred_internal.vtu")
                predicted_airfoils[0].save(destination / f"{simulation}_pred_aerofoil.vtp")

        if not true_coefficients:
            raise RuntimeError("No AirfRANS samples were evaluated")
        true_array = np.asarray(true_coefficients)
        predicted_array = np.asarray(predicted_coefficients)
        spearman = np.asarray(
            [
                scipy_stats.spearmanr(true_array[:, 0], predicted_array[:, 0])[0],
                scipy_stats.spearmanr(true_array[:, 1], predicted_array[:, 1])[0],
            ]
        )
        return {
            "model_name": model_name,
            "mean_time": np.asarray(elapsed_times).mean(axis=0),
            "std_time": np.asarray(elapsed_times).std(axis=0),
            "mean_score_vol": np.asarray(scores_volume).mean(axis=0),
            "mean_score_surf": np.asarray(scores_surface).mean(axis=0),
            "mean_score_force": np.asarray(scores_force).mean(axis=0),
            "mean_rel_p": np.asarray(scores_pressure).mean(axis=0),
            "mean_rel_wss": np.asarray(scores_wss).mean(axis=0),
            "spearman_coef": spearman,
        }

    @staticmethod
    def paper_metrics(score: Mapping[str, Any]) -> dict[str, float]:
        """从完整评估结果中提取论文表格对应的五项总体指标。"""
        volume = np.asarray(score["mean_score_vol"])
        surface = np.asarray(score["mean_score_surf"])
        force = np.asarray(score["mean_score_force"])
        spearman = np.asarray(score["spearman_coef"])
        return {
            "Volume": float(volume.mean()),
            "Surf": float(surface.mean()),
            "Surf_pressure": float(surface[2]),
            "CL_relative_error": float(force[1]),
            "rho_L": float(spearman[1]),
        }


__all__ = [
    "AirfRANSEvaluator",
    "json_ready",
    "seed_everything",
]
