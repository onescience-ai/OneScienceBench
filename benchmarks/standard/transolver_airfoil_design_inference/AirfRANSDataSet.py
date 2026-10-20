"""不依赖 OneScience 的 AirfRANS 数据集与数据加载器。

该文件仅依赖 NumPy、PyTorch、PyTorch Geometric 和 PyVista，既可被
``inference.py`` 导入，也可直接运行以检查一个真实 AirfRANS 样本。训练时
使用的归一化统计量直接定义在本文件中，不需要额外的 ``.npy`` 文件。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
import torch_geometric.nn as pyg_nn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader


NormTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# Training-set normalization statistics used by the published checkpoint.
MEAN_IN = np.asarray(
    [
        0.5006739497184753,
        0.011246345937252045,
        61.503807067871094,
        4.850043296813965,
        0.23172970116138458,
        -0.0016548282001167536,
        -0.00008646764035802335,
    ],
    dtype=np.float32,
)
STD_IN = np.asarray(
    [
        0.7040368914604187,
        0.33640000224113464,
        17.780105590820312,
        6.284955024719238,
        0.4752075970172882,
        0.04153645411133766,
        0.062408167868852615,
    ],
    dtype=np.float32,
)
MEAN_OUT = np.asarray(
    [
        42.43694305419922,
        9.696487426757812,
        -452.3697509765625,
        0.0007954502943903208,
    ],
    dtype=np.float32,
)
STD_OUT = np.asarray(
    [
        29.64150619506836,
        30.87746810913086,
        2963.272705078125,
        0.0029404046945273876,
    ],
    dtype=np.float32,
)
NORMALIZATION_STATS: NormTuple = (MEAN_IN, STD_IN, MEAN_OUT, STD_OUT)


def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    """按照目标点顺序重排物理量，替代 OneScience 中的同名辅助函数。"""
    index = np.empty(out_order_points.shape[0], dtype=np.int64)
    for i, point in enumerate(out_order_points):
        matches = np.all(in_order_points == point, axis=1)
        found = np.flatnonzero(matches)
        if found.size == 0:
            raise ValueError(f"Cannot match surface point {point.tolist()}")
        index[i] = found[0]
    return quantity_to_reordered[index]


def cell_sampling_2d(cell_points, cell_attr=None):
    """在二维四边形单元中随机采样，并对顶点属性做三角形插值。"""
    v0 = cell_points[:, 1] - cell_points[:, 0]
    v1 = cell_points[:, 3] - cell_points[:, 0]
    v2 = cell_points[:, 3] - cell_points[:, 2]
    v3 = cell_points[:, 1] - cell_points[:, 2]
    a0 = np.abs(np.linalg.det(np.hstack([v0[:, :2], v1[:, :2]]).reshape(-1, 2, 2)))
    a1 = np.abs(np.linalg.det(np.hstack([v2[:, :2], v3[:, :2]]).reshape(-1, 2, 2)))
    probability = a0 / (a0 + a1)
    triangle = np.random.binomial(1, probability)[:, None]
    uniform = np.random.uniform(size=(len(probability), 2))
    sampled = triangle * (uniform[:, 0:1] * v0 + uniform[:, 1:2] * v1) + (
        1 - triangle
    ) * (uniform[:, 0:1] * v2 + uniform[:, 1:2] * v3)
    mirrored = triangle * (
        (1 - uniform[:, 0:1]) * v0 + (1 - uniform[:, 1:2]) * v1
    ) + (1 - triangle) * (
        (1 - uniform[:, 0:1]) * v2 + (1 - uniform[:, 1:2]) * v3
    )
    sampled[uniform.sum(axis=1) > 1] = mirrored[uniform.sum(axis=1) > 1]

    sampled_attr = None
    if cell_attr is not None:
        t0 = np.zeros_like(v0)
        t1 = triangle * v0 + (1 - triangle) * v2
        t2 = triangle * v1 + (1 - triangle) * v3
        denominator = (
            (t1[:, 1] - t2[:, 1]) * (t0[:, 0] - t2[:, 0])
            + (t2[:, 0] - t1[:, 0]) * (t0[:, 1] - t2[:, 1])
        )
        w0 = (
            (t1[:, 1] - t2[:, 1]) * (sampled[:, 0] - t2[:, 0])
            + (t2[:, 0] - t1[:, 0]) * (sampled[:, 1] - t2[:, 1])
        ) / denominator
        w1 = (
            (t2[:, 1] - t0[:, 1]) * (sampled[:, 0] - t2[:, 0])
            + (t0[:, 0] - t2[:, 0]) * (sampled[:, 1] - t2[:, 1])
        ) / denominator
        w2 = 1 - w0 - w1
        if cell_attr.ndim == 2:
            cell_attr = cell_attr[:, :, None]
        attr0 = triangle * cell_attr[:, 0] + (1 - triangle) * cell_attr[:, 2]
        attr1 = cell_attr[:, 1]
        attr2 = cell_attr[:, 3]
        sampled_attr = w0[:, None] * attr0 + w1[:, None] * attr1 + w2[:, None] * attr2

    sampled += triangle * cell_points[:, 0] + (1 - triangle) * cell_points[:, 2]
    if sampled_attr is None:
        return sampled[:, :2]
    return np.hstack([sampled[:, :2], sampled_attr])


def cell_sampling_1d(line_points, line_attr=None):
    """在线段单元中随机采样，并线性插值端点属性。"""
    uniform = np.random.uniform(size=(len(line_points), 1))
    sampled = uniform * line_points[:, 0] + (1 - uniform) * line_points[:, 1]
    if line_attr is None:
        return sampled[:, :2]
    if line_attr.ndim == 2:
        line_attr = line_attr[:, :, None]
    sampled_attr = uniform * line_attr[:, 0] + (1 - uniform) * line_attr[:, 1]
    return np.hstack([sampled[:, :2], sampled_attr])


class AirfRANSDataset(Dataset):
    """从官方 VTU/VTP 文件读取 AirfRANS 图样本。"""

    def __init__(self, config: Any, mode: str = "train", coef_norm: Optional[NormTuple] = None):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Invalid mode: {mode}")
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(f"AirfRANSDataset.{mode}")
        self.logger.setLevel(logging.DEBUG if bool(getattr(config, "verbose", False)) else logging.INFO)
        self.data_path = Path(config.source.data_dir).expanduser()
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        self.data_list_names = self._read_split()
        self.coef_norm = self._normalization_stats(coef_norm)
        self.logger.info("[%s] AirfRANS dataset initialized with %d simulations.", mode, len(self))

    @staticmethod
    def _normalization_stats(coef_norm: Optional[NormTuple]) -> NormTuple:
        values = NORMALIZATION_STATS if coef_norm is None else coef_norm
        arrays = tuple(
            np.asarray(value, dtype=np.float32).copy() for value in values
        )
        shapes = tuple(value.shape for value in arrays)
        if shapes != ((7,), (7,), (4,), (4,)):
            raise ValueError(f"Invalid normalization statistic shapes: {shapes}")
        if not np.all(arrays[1] > 0) or not np.all(arrays[3] > 0):
            raise ValueError("Normalization standard deviations must be positive")
        return arrays  # type: ignore[return-value]

    def _read_split(self):
        manifest_path = self.data_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest.json not found at: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = self.config.data.splits
        if self.mode in {"train", "val"}:
            names = list(manifest[splits.train_name])
            n_val = int(len(names) * float(splits.val_split_ratio))
            if self.mode == "train":
                return names if n_val == 0 else names[:-n_val]
            return [] if n_val == 0 else names[-n_val:]
        return list(manifest[splits.test_name])

    def _load_single_simulation(self, sim_name: str):
        internal_path = self.data_path / sim_name / f"{sim_name}_internal.vtu"
        aerofoil_path = self.data_path / sim_name / f"{sim_name}_aerofoil.vtp"
        if not internal_path.is_file() or not aerofoil_path.is_file():
            raise FileNotFoundError(f"Incomplete AirfRANS sample: {sim_name}")
        internal = pv.read(internal_path).compute_cell_sizes(length=False, volume=False)
        aerofoil = pv.read(aerofoil_path)

        crop = self.config.data.crop
        if crop:
            internal = internal.clip_box((*crop, 0, 1), invert=False, crinkle=True)
        strategy = self.config.data.sampling.sample_strategy
        u_inf = float(sim_name.split("_")[2])
        alpha = float(sim_name.split("_")[3]) * np.pi / 180

        if strategy:
            return self._sample_cells(internal, aerofoil, u_inf, alpha, strategy)

        surf = internal.point_data["U"][:, 0] == 0
        geom = -internal.point_data["implicit_distance"][:, None]
        velocity = np.array([np.cos(alpha), np.sin(alpha)]).reshape(1, 2) * u_inf
        velocity = velocity * np.ones_like(internal.point_data["U"][:, :1])
        normal = np.zeros_like(velocity)
        normal[surf] = reorganize(
            aerofoil.points[:, :2],
            internal.points[surf, :2],
            -aerofoil.point_data["Normals"][:, :2],
        )
        pos = internal.points[:, :2]
        features = np.concatenate([pos, velocity, geom, normal], axis=1)
        target = np.concatenate(
            [
                internal.point_data["U"][:, :2],
                internal.point_data["p"][:, None],
                internal.point_data["nut"][:, None],
            ],
            axis=1,
        )
        return pos, features, target, surf

    def _sample_cells(self, internal, aerofoil, u_inf, alpha, strategy):
        sampling = self.config.data.sampling
        n_volume = int(sampling.n_boot)
        n_surface = int(n_volume * float(sampling.surf_ratio))
        if strategy == "uniform":
            cell_probability = internal.cell_data["Area"] / internal.cell_data["Area"].sum()
            line_probability = aerofoil.cell_data["Length"] / aerofoil.cell_data["Length"].sum()
            cell_indices = np.random.choice(internal.n_cells, n_volume, p=cell_probability)
            line_indices = np.random.choice(aerofoil.n_cells, n_surface, p=line_probability)
        elif strategy == "mesh":
            cell_indices = np.random.choice(internal.n_cells, n_volume)
            line_indices = np.random.choice(aerofoil.n_cells, n_surface)
        else:
            raise ValueError(f"Unknown sample strategy: {strategy}")

        cells = internal.cells.reshape(-1, 5)[cell_indices, 1:]
        cell_points = internal.points[cells]
        geom = -internal.point_data["implicit_distance"][cells, None]
        velocity = np.array([np.cos(alpha), np.sin(alpha)]).reshape(1, 1, 2) * u_inf
        velocity = velocity * np.ones_like(internal.point_data["U"][cells, :1])
        normal = np.zeros_like(velocity)
        attributes = np.concatenate(
            [
                velocity,
                geom,
                normal,
                internal.point_data["U"][cells, :2],
                internal.point_data["p"][cells, None],
                internal.point_data["nut"][cells, None],
            ],
            axis=-1,
        )
        volume = cell_sampling_2d(cell_points, attributes)
        pos_volume = volume[:, :2]
        x_volume = np.concatenate([pos_volume, volume[:, 2:7]], axis=1)
        y_volume = volume[:, 7:]

        lines = aerofoil.lines.reshape(-1, 3)[line_indices, 1:]
        line_points = aerofoil.points[lines]
        surf_velocity = np.array([np.cos(alpha), np.sin(alpha)]).reshape(1, 1, 2) * u_inf
        surf_velocity = surf_velocity * np.ones_like(aerofoil.point_data["U"][lines, :1])
        surf_attributes = np.concatenate(
            [
                surf_velocity,
                np.zeros_like(aerofoil.point_data["U"][lines, :1]),
                -aerofoil.point_data["Normals"][lines, :2],
                aerofoil.point_data["U"][lines, :2],
                aerofoil.point_data["p"][lines, None],
                aerofoil.point_data["nut"][lines, None],
            ],
            axis=-1,
        )
        surface = cell_sampling_1d(line_points, surf_attributes)
        pos_surface = surface[:, :2]
        x_surface = np.concatenate([pos_surface, surface[:, 2:7]], axis=1)
        y_surface = surface[:, 7:]
        return (
            np.concatenate([pos_volume, pos_surface]),
            np.concatenate([x_volume, x_surface]),
            np.concatenate([y_volume, y_surface]),
            np.concatenate(
                [np.zeros(len(pos_volume), dtype=bool), np.ones(len(pos_surface), dtype=bool)]
            ),
        )

    def __len__(self):
        return len(self.data_list_names)

    def __getitem__(self, index: int) -> Data:
        sim_name = self.data_list_names[index]
        pos, features, target, surf = self._load_single_simulation(sim_name)
        mean_in, std_in, mean_out, std_out = self.coef_norm
        features = (features - mean_in) / (std_in + 1e-8)
        target = (target - mean_out) / (std_out + 1e-8)
        pos_t = torch.as_tensor(pos, dtype=torch.float32)
        x_t = torch.as_tensor(features, dtype=torch.float32)
        y_t = torch.as_tensor(target, dtype=torch.float32)
        surf_t = torch.as_tensor(surf, dtype=torch.bool)

        if self.mode in {"train", "val"}:
            count = int(self.config.data.subsampling)
            if pos_t.size(0) > count:
                selected = torch.tensor(random.sample(range(pos_t.size(0)), count))
                pos_t, x_t, y_t, surf_t = (
                    pos_t[selected],
                    x_t[selected],
                    y_t[selected],
                    surf_t[selected],
                )

        model_hparams = getattr(self.config, "model_hparams", None)
        build_graph = bool(getattr(model_hparams, "build_graph", False))
        edge_index = None
        if build_graph:
            edge_index = pyg_nn.radius_graph(
                x=pos_t,
                r=float(model_hparams.r),
                loop=True,
                max_num_neighbors=int(model_hparams.max_neighbors),
            )
        return Data(pos=pos_t, x=x_t, y=y_t, surf=surf_t, edge_index=edge_index)


# 同时保留用户指定文件名风格的类名别名。
AirfRANSDataSet = AirfRANSDataset


class AirfRANSDatapipe:
    """使用 PyTorch/PyG 原生组件创建训练、验证和测试加载器。"""

    def __init__(self, params: Any, distributed: bool = False):
        self.params = params
        self.distributed = bool(distributed)
        self.train_dataset = AirfRANSDataset(params, mode="train")
        self.coef_norm = self.train_dataset.coef_norm
        self.val_dataset = AirfRANSDataset(params, mode="val", coef_norm=self.coef_norm)
        self.test_dataset = AirfRANSDataset(params, mode="test", coef_norm=self.coef_norm)

    def _loader(self, dataset, shuffle: bool, drop_last: bool = False):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
        loader = PyGDataLoader(
            dataset,
            batch_size=int(self.params.dataloader.batch_size),
            drop_last=drop_last and self.distributed,
            num_workers=int(self.params.dataloader.num_workers),
            pin_memory=torch.cuda.is_available(),
            shuffle=shuffle and sampler is None,
            sampler=sampler,
        )
        return loader, sampler

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)[0]


def _standalone_config(args):
    """为命令行自检创建不依赖外部配置类的简单配置对象。"""
    return SimpleNamespace(
        verbose=True,
        source=SimpleNamespace(data_dir=args.data_dir),
        data=SimpleNamespace(
            splits=SimpleNamespace(
                train_name="full_train",
                test_name="full_test",
                val_split_ratio=0.1,
            ),
            sampling=SimpleNamespace(sample_strategy=None, n_boot=500000, surf_ratio=0.1),
            crop=None,
            subsampling=32000,
        ),
        dataloader=SimpleNamespace(batch_size=1, num_workers=0),
        model_hparams=SimpleNamespace(build_graph=False, r=0.05, max_neighbors=64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone AirfRANS dataset smoke test")
    parser.add_argument("--data-dir", required=True, help="Directory containing manifest.json")
    parser.add_argument("--mode", choices=("train", "val", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    dataset = AirfRANSDataset(_standalone_config(args), mode=args.mode)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index {args.index} is outside dataset length {len(dataset)}")
    sample = dataset[args.index]
    print(f"dataset_length={len(dataset)}")
    print(f"simulation={dataset.data_list_names[args.index]}")
    print(f"pos={tuple(sample.pos.shape)}, x={tuple(sample.x.shape)}, y={tuple(sample.y.shape)}")
    print(f"surface_points={int(sample.surf.sum())}, edge_index={sample.edge_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AirfRANSDataSet",
    "AirfRANSDataset",
    "AirfRANSDatapipe",
    "MEAN_IN",
    "MEAN_OUT",
    "NORMALIZATION_STATS",
    "STD_IN",
    "STD_OUT",
]
