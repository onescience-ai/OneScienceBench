"""Standalone compact MACE model, PyTorch Dataset, and training launcher reference."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from ase.io import read
from torch import nn
from torch.utils.data import DataLoader, Dataset


class MACEModel(nn.Module):
    """紧凑的 MACE 风格消息传递能量模型，输入原子序数与坐标，输出结构能量。"""

    def __init__(self, atomic_numbers: Sequence[int] = (1, 8), hidden_dim: int = 16,
                 num_radial: int = 8, num_interactions: int = 2, r_max: float = 4.0) -> None:
        """初始化元素嵌入、径向基函数、消息传递层与能量预测头。"""
        super().__init__()
        if not atomic_numbers or hidden_dim < 1 or num_radial < 1 or num_interactions < 1 or r_max <= 0:
            raise ValueError("invalid MACE dimensions or cutoff")
        self.atomic_numbers = tuple(int(z) for z in atomic_numbers)
        self.hidden_dim, self.num_radial = int(hidden_dim), int(num_radial)
        self.num_interactions, self.r_max = int(num_interactions), float(r_max)
        self.embedding = nn.Embedding(max(self.atomic_numbers) + 1, self.hidden_dim)
        self.register_buffer("radial_centers", torch.linspace(0.0, self.r_max, self.num_radial))
        self.register_buffer("radial_width", torch.tensor(float(self.num_radial) / self.r_max))
        self.radial_projection = nn.Sequential(nn.Linear(self.num_radial, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.message_layers = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
            for _ in range(self.num_interactions)
        )
        self.energy_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, 1))

    def _radial_features(self, distances: torch.Tensor) -> torch.Tensor:
        """根据原子间距离计算高斯径向基与余弦截断函数的乘积。"""
        scaled = distances.unsqueeze(-1) - self.radial_centers
        basis = torch.exp(-self.radial_width.square() * scaled.square())
        cutoff = 0.5 * (torch.cos(torch.pi * distances / self.r_max) + 1.0)
        return basis * cutoff.clamp_min(0.0).unsqueeze(-1)

    def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor,
                atom_mask: torch.Tensor | None = None) -> torch.Tensor:
        """执行消息传递前向计算，返回每个结构的标量能量。"""
        if atomic_numbers.ndim != 2 or positions.ndim != 3 or positions.shape[:2] != atomic_numbers.shape or positions.shape[-1] != 3:
            raise ValueError("atomic_numbers must be (B,N), positions must be (B,N,3)")
        if atom_mask is None:
            atom_mask = atomic_numbers > 0
        atom_mask = atom_mask.to(device=positions.device, dtype=torch.bool)
        if torch.any(atomic_numbers[atom_mask] >= self.embedding.num_embeddings):
            raise ValueError("atomic number exceeds model element table")
        node_state = self.embedding(atomic_numbers.clamp_min(0))
        displacement = positions[:, :, None, :] - positions[:, None, :, :]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
        eye = torch.eye(atomic_numbers.shape[1], device=positions.device, dtype=torch.bool).unsqueeze(0)
        radial_state = self.radial_projection(self._radial_features(distances))
        radial_state = radial_state * (pair_mask & ~eye & (distances < self.r_max)).unsqueeze(-1)
        for layer in self.message_layers:
            messages = (node_state[:, None, :, :] * radial_state).sum(dim=2)
            node_state = (node_state + layer(torch.cat((node_state, messages), dim=-1))) * atom_mask.unsqueeze(-1)
        return (self.energy_head(node_state).squeeze(-1) * atom_mask).sum(dim=1)


def _value(group: h5py.Group, name: str) -> Any:
    """从 HDF5 组中读取指定字段，若值为 b'None' 则返回 None。"""
    if name in group:
        value = group[name][()]
        return None if isinstance(value, bytes) and value == b"None" else value
    properties = group.get("properties")
    if isinstance(properties, h5py.Group) and name in properties:
        return properties[name][()]
    return None


def _read_hdf5(path: Path, energy_key: str, forces_key: str) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """读取 HDF5 文件或目录，返回 (atomic_numbers, positions, energy, forces) 样本列表。"""
    samples: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
    files = sorted(path.glob("*.h5")) if path.is_dir() else [path]
    for filename in files:
        with h5py.File(filename, "r") as handle:
            groups: list[h5py.Group] = []
            handle.visititems(lambda _, obj: groups.append(obj) if isinstance(obj, h5py.Group) and ("atomic_numbers" in obj or "node_attrs" in obj) else None)
            for group in groups:
                numbers = np.asarray(_value(group, "atomic_numbers") if "atomic_numbers" in group else _value(group, "node_attrs"), dtype=np.int64).reshape(-1)
                positions = np.asarray(_value(group, "positions"), dtype=np.float32)
                energy, forces = _value(group, energy_key), _value(group, forces_key)
                if energy is None or forces is None:
                    raise KeyError(f"missing {energy_key!r} or {forces_key!r} in {filename}")
                samples.append((numbers, positions, float(np.asarray(energy).reshape(-1)[0]), np.asarray(forces, dtype=np.float32)))
    if not samples:
        raise ValueError(f"no energy/forces configurations found in {path}")
    return samples


def _read_extxyz(path: Path, energy_key: str, forces_key: str) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """读取 extxyz 文件，返回 (atomic_numbers, positions, energy, forces) 样本列表。"""
    frames = read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    result = []
    for atoms in frames:
        if energy_key not in atoms.info or forces_key not in atoms.arrays:
            raise KeyError(f"missing {energy_key!r} or {forces_key!r} in {path}")
        result.append((atoms.get_atomic_numbers().astype(np.int64), atoms.get_positions().astype(np.float32), float(atoms.info[energy_key]), np.asarray(atoms.arrays[forces_key], dtype=np.float32)))
    return result


def _read_data(path: Path, data_format: str, energy_key: str, forces_key: str) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """根据数据格式选择对应的读取函数。"""
    return _read_hdf5(path, energy_key, forces_key) if data_format == "hdf5" else _read_extxyz(path, energy_key, forces_key)


class MACEDataset(Dataset):
    """将 extxyz 或 HDF5 格式的 MACE 训练数据封装为 PyTorch Dataset。"""

    def __init__(self, data_path: str | Path, data_format: str,
                 energy_key: str, forces_key: str,
                 atomic_numbers: Sequence[int] | None = None,
                 statistics_file: str | Path | None = None,
                 r_max: float = 4.0) -> None:
        """初始化数据集，读取所有样本到内存。"""
        if data_format not in {"extxyz", "hdf5"}:
            raise ValueError("data_format must be 'extxyz' or 'hdf5'")
        self.data_format = data_format
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.r_max = float(r_max)
        self.atomic_numbers = tuple(int(z) for z in atomic_numbers) if atomic_numbers else None
        self.statistics_file = Path(statistics_file).resolve() if statistics_file else None
        self.samples = _read_data(Path(data_path), data_format, energy_key, forces_key)

    def __len__(self) -> int:
        """返回数据集中样本数量。"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """返回指定索引的样本并统一数据类型。"""
        numbers, positions, energy, forces = self.samples[idx]
        return (
            np.asarray(numbers, dtype=np.int64),
            np.asarray(positions, dtype=np.float32),
            float(energy),
            np.asarray(forces, dtype=np.float32),
        )


def mace_collate_fn(batch: Sequence[tuple[np.ndarray, np.ndarray, float, np.ndarray]]) -> tuple[torch.Tensor, ...]:
    """将不同原子数目的样本填充为等长张量，并生成原子有效掩码。"""
    max_n = max(sample[0].shape[0] for sample in batch)
    b = len(batch)
    atomic_numbers = torch.zeros(b, max_n, dtype=torch.long)
    positions = torch.zeros(b, max_n, 3, dtype=torch.float32)
    forces = torch.zeros(b, max_n, 3, dtype=torch.float32)
    atom_mask = torch.zeros(b, max_n, dtype=torch.bool)
    energy = torch.zeros(b, dtype=torch.float32)
    for i, (numbers, pos, e, f) in enumerate(batch):
        n = numbers.shape[0]
        atomic_numbers[i, :n] = torch.from_numpy(numbers)
        positions[i, :n] = torch.from_numpy(pos)
        forces[i, :n] = torch.from_numpy(f)
        atom_mask[i, :n] = True
        energy[i] = float(e)
    return atomic_numbers, positions, energy, forces, atom_mask


class MACETrainLauncher:
    """读取用户数据，训练 MACEModel，并保存 checkpoint、日志与指标。"""

    def __init__(self, train_file: str, output_dir: str, valid_file: str | None = None, test_file: str | None = None,
                 epochs: int = 10, device: str = "cpu", seed: int = 123, energy_key: str = "REF_energy",
                 forces_key: str = "REF_forces", e0s: str = "average", data_format: str = "extxyz",
                 statistics_file: str | None = None, atomic_numbers: Sequence[int] | None = None, r_max: float = 4.0,
                 batch_size: int = 2, valid_batch_size: int = 2, num_channels: int = 16, max_l: int = 1,
                 num_interactions: int = 2, lr: float = 1e-3) -> None:
        """保存训练配置并校验参数合法性。"""
        if data_format not in {"extxyz", "hdf5"} or epochs < 1 or r_max <= 0 or batch_size < 1 or valid_batch_size < 1 or lr <= 0:
            raise ValueError("invalid training configuration")
        if data_format == "hdf5" and atomic_numbers is None and statistics_file is None:
            raise ValueError("HDF5 input requires atomic_numbers or statistics_file")
        self.train_file, self.valid_file, self.test_file = Path(train_file).resolve(), Path(valid_file).resolve() if valid_file else None, Path(test_file).resolve() if test_file else None
        self.output_dir, self.epochs, self.device, self.seed = Path(output_dir).resolve(), epochs, device, seed
        self.energy_key, self.forces_key, self.e0s, self.data_format = energy_key, forces_key, e0s, data_format
        self.statistics_file, self.atomic_numbers = Path(statistics_file).resolve() if statistics_file else None, tuple(atomic_numbers) if atomic_numbers else None
        self.r_max, self.batch_size, self.valid_batch_size, self.num_channels, self.max_l, self.num_interactions, self.lr = r_max, batch_size, valid_batch_size, num_channels, max_l, num_interactions, lr

    def _build_loader(self, path: Path | None, batch_size: int, shuffle: bool) -> DataLoader | None:
        """为指定数据路径构建 DataLoader。"""
        if path is None:
            return None
        dataset = MACEDataset(path, self.data_format, self.energy_key, self.forces_key,
                              atomic_numbers=self.atomic_numbers, statistics_file=self.statistics_file, r_max=self.r_max)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=mace_collate_fn)

    def _loss(self, model: MACEModel, atomic_numbers: torch.Tensor, positions: torch.Tensor,
              energy: torch.Tensor, forces: torch.Tensor, atom_mask: torch.Tensor,
              training: bool) -> torch.Tensor:
        """计算预测能量与力相对于标签的联合均方误差损失。"""
        positions = positions.to(self._device).requires_grad_(True)
        atomic_numbers = atomic_numbers.to(self._device)
        atom_mask = atom_mask.to(self._device)
        pred_energy = model(atomic_numbers, positions, atom_mask)
        pred_forces = -torch.autograd.grad(pred_energy.sum(), positions,
                                           create_graph=training, retain_graph=training)[0]
        energy_loss = (pred_energy - energy.to(self._device)).square().mean()
        forces_mask = atom_mask.unsqueeze(-1)
        forces_loss = ((pred_forces - forces.to(self._device)).square() * forces_mask).sum() / forces_mask.sum().clamp_min(1.0)
        return energy_loss + forces_loss

    def run(self) -> Path:
        """执行训练循环，保存模型、指标与训练清单，返回输出目录。"""
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self._device = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        train_loader = self._build_loader(self.train_file, self.batch_size, shuffle=True)
        valid_loader = self._build_loader(self.valid_file, self.valid_batch_size, shuffle=False) if self.valid_file else train_loader
        if train_loader is None:
            raise ValueError("train_file is required")
        elements = self.atomic_numbers or tuple(sorted({int(z) for sample in train_loader.dataset.samples for z in sample[0]}))
        model = MACEModel(elements, hidden_dim=self.num_channels, num_interactions=self.num_interactions, r_max=self.r_max).to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics: list[dict[str, float | int]] = []

        def evaluate() -> float:
            """在验证集上计算平均损失。"""
            model.eval()
            values: list[float] = []
            for batch in valid_loader:
                with torch.enable_grad():
                    values.append(float(self._loss(model, *batch, training=False).detach().cpu()))
            return float(np.mean(values)) if values else float("nan")

        with (self.output_dir / "train.log").open("w", encoding="utf-8") as log:
            initial = evaluate()
            metrics.append({"step": 0, "epoch": -1, "loss": initial})
            log.write(f"Initial: loss={initial:.8f}\n")
            for epoch in range(self.epochs):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = self._loss(model, *batch, training=True)
                    loss.backward()
                    optimizer.step()
                value = evaluate()
                metrics.append({"step": len(metrics), "epoch": epoch, "loss": value})
                log.write(f"Epoch {epoch}: loss={value:.8f}\n")
            log.write("Training complete\n")
        checkpoint = {"model_state_dict": model.cpu().state_dict(), "model_config": {"atomic_numbers": elements, "hidden_dim": self.num_channels, "num_radial": 8, "num_interactions": self.num_interactions, "r_max": self.r_max}, "epoch": self.epochs}
        torch.save(checkpoint, self.output_dir / "mace_train.model")
        (self.output_dir / "metrics.jsonl").write_text("\n".join(json.dumps(point) for point in metrics) + "\n", encoding="utf-8")
        (self.output_dir / "training_manifest.json").write_text(json.dumps({"data_format": self.data_format, "train_file": str(self.train_file), "valid_file": str(self.valid_file) if self.valid_file else None, "model": "MACEModel", "epochs": self.epochs}, indent=2), encoding="utf-8")
        return self.output_dir
