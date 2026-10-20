################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn.functional as F
from ase.data import (  # reference_states, # stored lattice constant, bond length, etc.
    atomic_masses,
    atomic_names,
    atomic_numbers,
    chemical_symbols,
    covalent_radii,
    ground_state_magnetic_moments,
    vdw_radii,
)
from torch import Tensor

ATOMIC_SYMBOLS: List[str]
ATOMIC_NUMBERS: Dict[str, int]
ATOMIC_NAMES: List[str]
ATOMIC_MASSES: List[float]
ATOMIC_MAGMOM: List[float]
ATOMIC_VDW_RADII: List[float]
ATOMIC_COVALENT_RADII: List[float]

ATOMIC_SYMBOLS = chemical_symbols
ATOMIC_NUMBERS = atomic_numbers
ATOMIC_NAMES = atomic_names
ATOMIC_NAMES[0] = "Dummy"
ATOMIC_MASSES = atomic_masses.tolist()
ATOMIC_VDW_RADII = vdw_radii.tolist()
ATOMIC_COVALENT_RADII = (
    covalent_radii.tolist()
)  # if covalent_radii = 0.2, that means this element missing covalent_radii in ase
ATOMIC_MAGMOM = ground_state_magnetic_moments.tolist()  # ground state


class TorchElement:
    def __init__(self, zs: Union[List[int], Tuple[int, ...]]):

        self.zs = sorted(list(zs))
        self.num_elements = len(zs)

        self.lookup_table = torch.full((max(self.zs) + 1,), -1, dtype=torch.int64)
        for idx, z in enumerate(self.zs):
            self.lookup_table[z] = idx

    def z2idx(self, z: Tensor) -> Tensor:
        return self.lookup_table[z]

    def z2onehot(self, zs: Tensor) -> Tensor:
        """
        :param zs: shape (N,),
        :return: shape (N, num_classes), one-hot
        """
        lookup_table = self.lookup_table.to(zs.device)
        idxs = lookup_table[zs]  # shape (N,)
        return F.one_hot(idxs, num_classes=self.num_elements)

    def __len__(self) -> int:
        return len(self.zs)


def build_element_lookup(atomic_numbers: Sequence[int]) -> TorchElement:
    element = TorchElement(sorted(list(atomic_numbers)))
    return element
