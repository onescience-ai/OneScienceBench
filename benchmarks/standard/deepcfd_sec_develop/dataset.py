import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


def _load_pickle(path):
    with open(path, "rb") as data_file:
        return np.asarray(pickle.load(data_file), dtype=np.float32)


def create_dataloaders(
    directory,
    input_file="dataX.pkl",
    target_file="dataY.pkl",
    split_ratio=0.7,
    seed=0,
    batch_size=16,
    num_workers=0,
):
    """Load DeepCFD arrays and create deterministic train/test loaders."""
    directory = Path(directory)
    inputs = _load_pickle(directory / input_file)
    targets = _load_pickle(directory / target_file)

    if len(inputs) != len(targets):
        raise ValueError(
            f"Input and target sample counts differ: {len(inputs)} != {len(targets)}"
        )
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")

    indices = list(range(len(inputs)))
    random.Random(seed).shuffle(indices)
    split_index = int(len(indices) * split_ratio)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    dataset = TensorDataset(inputs, targets)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    train_targets = targets[train_indices]
    loss_weights = torch.sqrt(torch.mean(train_targets**2, dim=(0, 2, 3)))
    return train_loader, test_loader, loss_weights.view(-1, 1, 1)
