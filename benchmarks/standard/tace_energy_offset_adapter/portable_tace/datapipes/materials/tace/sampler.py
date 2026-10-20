# TODO not for users now
################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import heapq
import math
from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Dataset, Sampler
from torch_geometric.data import Data


class EdgeBalancedBatchSampler(BatchSampler):
    """
    Batch sampler for Graph Datasets that balances batches by total number of edges.

    Features:
    - Builds global batches by max_edges_per_batch.
    - Splits each batch across GPUs using Largest Processing Time (LPT) greedy algorithm.
    - Supports single-GPU (non-DDP) and multi-GPU (DDP).
    """

    def __init__(
        self,
        dataset: Dataset,
        max_edges_per_batch: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.max_edges_per_batch = int(max_edges_per_batch)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # --- handle single GPU vs DDP ---
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1  # single GPU fallback

        self.epoch = 0

        # Lazy cache for edge counts
        self._edge_counts_cache: List[Optional[int]] = [None] * len(dataset)

    def set_epoch(self, epoch: int):
        """Set current training epoch (affects shuffle)."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # Build global batches by max_edges_per_batch
        global_batches = self._build_global_batches(indices)

        for batch_indices in global_batches:
            # Partition each global batch across GPUs using LPT
            per_rank_batches = self._partition_lpt(batch_indices)

            local_batch = per_rank_batches[self.rank]

            if self.drop_last and len(local_batch) == 0:
                continue

            yield local_batch

    def __len__(self) -> int:
        total_edges = sum(self._edge_count(i) for i in range(len(self.dataset)))
        return math.ceil(total_edges / self.max_edges_per_batch)

    def _edge_count(self, idx: int) -> int:
        """Lazy compute edge count with caching."""
        if self._edge_counts_cache[idx] is None:
            graph = self.dataset[idx]
            if hasattr(graph, "edge_index"):
                w = int(graph.edge_index.size(1))
            elif hasattr(graph, "num_edges"):
                w = int(graph.num_edges)
            else:
                raise AttributeError(
                    f"Sample {idx} has no `edge_index` or `num_edges` attribute"
                )
            self._edge_counts_cache[idx] = w
        return self._edge_counts_cache[idx]

    def _build_global_batches(self, indices: List[int]) -> List[List[int]]:
        """Accumulate samples into global batches by max_edges_per_batch."""
        batches = []
        current = []
        current_edges = 0

        for idx in indices:
            w = self._edge_count(idx)

            # handle huge graph
            if w > self.max_edges_per_batch:
                if not self.drop_last:
                    batches.append([idx])
                continue

            if current_edges + w > self.max_edges_per_batch:
                if current or not self.drop_last:
                    batches.append(current)
                current = []
                current_edges = 0

            current.append(idx)
            current_edges += w

        if current and not self.drop_last:
            batches.append(current)

        return batches

    def _partition_lpt(self, batch_indices: List[int]) -> List[List[int]]:
        """Partition a global batch into per-rank sub-batches using LPT greedy algorithm."""
        batch_indices_sorted = sorted(
            batch_indices, key=lambda i: self._edge_count(i), reverse=True
        )

        # Heap: (load, rank_index)
        heap = [(0, i) for i in range(self.world_size)]
        heapq.heapify(heap)

        buckets: List[List[int]] = [[] for _ in range(self.world_size)]
        for idx in batch_indices_sorted:
            load, rank_i = heapq.heappop(heap)
            buckets[rank_i].append(idx)
            load += self._edge_count(idx)
            heapq.heappush(heap, (load, rank_i))

        return buckets


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
