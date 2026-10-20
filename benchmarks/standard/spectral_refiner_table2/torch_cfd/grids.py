# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright (C) 2025 S.Cao
# ported Google's Jax-CFD functional template to torch.Tensor operations

from __future__ import annotations

import dataclasses
import math
import numbers
import operator
from functools import reduce

from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.fft as fft
import torch.nn.functional as F
from einops import rearrange as _rearrange, repeat as _repeat

from torch_cfd import tensor_utils

_HANDLED_TYPES = (numbers.Number, torch.Tensor)
T = TypeVar("T")  # for GridVariable vector


class BCType:
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    NONE = None


BCValue = Union[torch.Tensor, None]
BCValues = Tuple[BCValue, BCValue] | Sequence[Tuple[BCValue, BCValue]]


class Padding:
    MIRROR = "reflect"
    EXTEND = "replicate"
    SYMMETRIC = "symmetric"
    NONE = ""


@dataclasses.dataclass(init=False, frozen=True)
class Grid:
    """
    Describes the size and shape for an Arakawa C-Grid.
    Ported from jax_cfd.base.grids.Grid to pytorch

    See https://en.wikipedia.org/wiki/Arakawa_grids.

    This class describes domains that can be written as an outer-product of 1D
    grids. Along each dimension `i`:
    - `shape[i]` gives the whole number of grid cells on a single device.
    - `step[i]` is the width of each grid cell.
    - `(lower, upper) = domain[i]` gives the locations of lower and upper
      boundaries. The identity `upper - lower = step[i] * shape[i]` is enforced.

    Args:
        shape: (nx, ny)
        step: (dx, dy) or a single float for isotropic grids.
        domain: ((x0, x1), (y0, y1)), by default if only step
        is given the domain ((0, 1), (0, 1)).
        device: the device of the output grid.mesh().
    """

    shape: Tuple[int, ...]
    step: Tuple[float, ...]
    domain: Tuple[Tuple[float, float], ...]
    device: torch.device

    def __init__(
        self,
        shape: Sequence[int],
        step: Optional[Union[float, Sequence[float]]] = None,
        domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        """Construct a grid object."""
        shape = tuple(operator.index(s) for s in shape)
        object.__setattr__(self, "shape", shape)

        if step is not None and domain is not None:
            raise TypeError("cannot provide both step and domain")
        elif domain is not None:
            if isinstance(domain, (int, float)):
                domain = ((0, domain),) * len(shape)
            else:
                if len(domain) != self.ndim:
                    raise ValueError(
                        "length of domain does not match ndim: "
                        f"{len(domain)} != {self.ndim}"
                    )
                for bounds in domain:
                    if len(bounds) != 2:
                        raise ValueError(
                            f"domain is not sequence of pairs of numbers: {domain}"
                        )
            domain = tuple((float(lower), float(upper)) for lower, upper in domain)
        else:
            if step is None:
                step = 1.0
            if isinstance(step, numbers.Number):
                step = (step,) * self.ndim
            elif len(step) != self.ndim:
                raise ValueError(
                    "length of step does not match ndim: " f"{len(step)} != {self.ndim}"
                )
            domain = tuple(
                (0.0, float(step_ * size)) for step_, size in zip(step, shape)
            )

        object.__setattr__(self, "domain", domain)

        step = tuple(
            (upper - lower) / size for (lower, upper), size in zip(domain, shape)
        )
        object.__setattr__(self, "step", step)
        if device is None:
            device = torch.device("cpu")
        object.__setattr__(self, "device", device)

    def __repr__(self) -> str:
        lines = [f"Grid({self.ndim}D):"]
        lines.append(f"  shape: {self.shape}")

        for i in range(self.ndim):
            lower, upper = self.domain[i]
            step = self.step[i]
            lines.append(
                f"  dim {i}: [{lower:.3f}, {upper:.3f}], step={step:.3f}, size={self.shape[i]}"
            )

        lines.append(f"  device: {self.device}")

        return "\n".join(lines)

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of this grid."""
        return len(self.shape)

    @property
    def cell_center(self) -> Tuple[float, ...]:
        """Offset at the center of each grid cell."""
        return self.ndim * (0.5,)

    @property
    def cell_faces(self) -> Tuple[Tuple[float, ...], ...]:
        """Returns the offsets at each of the 'forward' cell faces."""
        d = self.ndim
        offsets = (torch.eye(d) + torch.ones([d, d])) / 2.0
        return tuple(tuple(float(o) for o in offset) for offset in offsets)

    def stagger(self, v: Tuple[torch.Tensor, ...]) -> Tuple[GridVariable, ...]:
        """Places the velocity components of `v` on the `Grid`'s cell faces."""
        offsets = self.cell_faces
        return GridVariableVector(
            tuple(GridVariable(u, o, self) for u, o in zip(v, offsets))
        )

    def center(self, v: Tuple[torch.Tensor, ...]) -> Tuple[GridVariable, ...]:
        """Places all arrays in the pytree `v` at the `Grid`'s cell center."""
        offset = self.cell_center
        return GridVariableVector(
            tuple(GridVariable(tensor, offset, self) for tensor in v)
        )

    def axes(
        self, offset: Optional[Sequence[float]] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Returns a tuple of arrays containing the grid points along each axis.

        Args:
          offset: an optional sequence of length `ndim`. The grid will be shifted by
            `offset * self.step`.

        Returns:
          An tuple of `self.ndim` arrays. The jth return value has shape
          `[self.shape[j]]`.
        """
        if offset is None:
            offset = self.cell_center
        if len(offset) != self.ndim:
            raise ValueError(
                f"unexpected offset length: {len(offset)} vs " f"{self.ndim}"
            )
        return tuple(
            lower + (torch.arange(length) + offset_i) * step
            for (lower, _), offset_i, length, step in zip(
                self.domain, offset, self.shape, self.step
            )
        )

    def fft_axes(self) -> Tuple[torch.Tensor, ...]:
        """Returns the ordinal frequencies corresponding to the axes.

        Transforms each axis into the *ordinal* frequencies for the Fast Fourier
        Transform (FFT). Multiply by `2 * jnp.pi` to get angular frequencies.

        Returns:
          A tuple of `self.ndim` arrays. The jth return value has shape
          `[self.shape[j]]`.
        """
        freq_axes = tuple(fft.fftfreq(n, d=s) for (n, s) in zip(self.shape, self.step))
        return freq_axes

    def mesh(
        self,
        offset: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Returns an tuple of arrays containing positions in each grid cell.

        Args:
          offset: an optional sequence of length `ndim`. The grid will be shifted by
            `offset * self.step`.

        Returns:
          An tuple of `self.ndim` arrays, each of shape `self.shape`. In 3
          dimensions, entry `self.mesh[n][i, j, k]` is the location of point
          `i, j, k` in dimension `n`.
        """
        axes = self.axes(offset)
        mesh = torch.meshgrid(*axes, indexing="ij")
        return tuple(xi.to(self.device) for xi in mesh)

    def fft_mesh(self) -> Tuple[torch.Tensor, ...]:
        """Returns a tuple of arrays containing positions in Fourier space."""
        fft_axes = self.fft_axes()
        fmesh = torch.meshgrid(*fft_axes, indexing="ij")
        return tuple(xi.to(self.device) for xi in fmesh)

    def rfft_mesh(self) -> Tuple[torch.Tensor, ...]:
        """Returns a tuple of arrays containing positions in rfft space."""
        fmesh = self.fft_mesh()
        k_max = math.floor(self.shape[-1] / 2.0)
        return tuple(xi[..., : k_max + 1] for xi in fmesh)

    def eval_on_mesh(
        self,
        fn: Callable[..., torch.Tensor],
        offset: Optional[Tuple[float, ...]] = None,
        bc: Optional[BoundaryConditions] = None,
    ) -> GridVariable:
        """Evaluates the function on the grid mesh with the specified offset.

        Args:
          fn: A function that accepts the mesh arrays and returns an array.
          offset: an optional sequence of length `ndim`.  If not specified, uses the offset for the cell center.
          bc: optional boundary conditions to wrap the variable with.

        Returns:
          fn(x, y, ...) evaluated on the mesh, as a GridArray with specified offset.

        Example:
        >>> f = lambda x, y: x + 2 * y
        >>> grid = Grid((4, 4), domain=((0.0, 1.0), (0.0, 1.0)))
        >>> offset = (0, 0)
        >>> u = grid.eval_on_mesh(f, offset)
        """
        if offset is None:
            offset = self.cell_center
        return GridVariable(fn(*self.mesh(offset)), offset, self, bc)

    def boundary_mesh(
        self,
        dim: int,
        offset: Optional[Tuple[float, ...]] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Get coordinate arrays for boundary points along dimension dim.

        Args:
            dim: The dimension along which to get boundary coordinates
            offset: Grid offset for coordinate calculation

        Returns:
            A tuple of (lower_boundary_coords, upper_boundary_coords) where each
            contains coordinate arrays for the boundary points.
            - 1D case: returns ((domain[0],), (domain[1],)) - scalar boundary points
            - 2D case:
              * dim=0 (x boundaries): ((x_left, y_coords), (x_right, y_coords))
                dim 0 corresponds to u[0, :] and u[-1, :]
              * dim=1 (y boundaries): ((x_coords, y_left), (x_coords, y_right))
                dim 1 corresponds to u[:, 0] and u[:, -1]
        """
        if offset is None:
            offset = self.cell_center

        # Handle 1D case
        if self.ndim == 1:
            lower_boundary = torch.tensor(self.domain[0][0], device=self.device)
            upper_boundary = torch.tensor(self.domain[0][1], device=self.device)
            return ((lower_boundary,), (upper_boundary,))

        # Handle 2D case
        elif self.ndim == 2:
            if dim < 0:
                dim = self.ndim + dim

            if dim not in (0, 1):
                raise ValueError(f"dim must be 0 or 1 for 2D grids, got {dim}")

            # Use XOR to determine which dimension varies along the boundary
            other_dim = dim ^ 1  # 0 ^ 1 = 1, 1 ^ 1 = 0

            # Get coordinates for the varying dimension (same for both boundaries)
            bd_varying_coords = (
                self.domain[other_dim][0]
                + (torch.arange(self.shape[other_dim], device=self.device) + offset[other_dim])
                * self.step[other_dim]
            )

            # Get boundary coordinates for the fixed dimension
            lower_coord = self.domain[dim][0]
            upper_coord = self.domain[dim][1]

            # Create coordinate arrays for boundaries
            lower_fixed_coords = torch.full_like(bd_varying_coords, lower_coord)
            upper_fixed_coords = torch.full_like(bd_varying_coords, upper_coord)

            # Arrange coordinates in proper order based on dimension
            if dim == 0:  # x boundaries
                return ((lower_fixed_coords, bd_varying_coords), (upper_fixed_coords, bd_varying_coords))
            else:  # dim == 1, y boundaries
                return ((bd_varying_coords, lower_fixed_coords), (bd_varying_coords, upper_fixed_coords))

        else:
            raise NotImplementedError(
                f"boundary_mesh not implemented for {self.ndim}D grids"
            )


@dataclasses.dataclass(init=False, frozen=True)
class BoundaryConditions:
    """Base class for boundary conditions on a PDE variable.

    Attributes:
      types: `types[i]` is a tuple specifying the lower and upper BC types for
        dimension `i`.
    """

    types: Tuple[Tuple[str, str], ...]
    bc_values: Union[float, BCValue, Tuple[Tuple[BCValue, BCValue], ...]]
    ndim: int

    def shift(
        self,
        u: GridVariable,
        offset: int,
        dim: int,
    ) -> GridVariable:
        """Shift an GridVariable by `offset`.

        Args:
          u: an `GridVariable` object.
          offset: positive or negative integer offset to shift.
          dim: axis to shift along.

        Returns:
          A copy of `u`, shifted by `offset`. The returned `GridVariable` has offset
          `u.offset + offset`.
        """
        raise NotImplementedError(
            "shift() not implemented in BoundaryConditions base class."
        )

    def values(
        self,
        dim: int,
        grid: Grid,
        offset: Optional[Tuple[float, ...]],
        time: Optional[float],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns torch.Tensors specifying boundary values on the grid along axis.

        Args:
          dim: axis along which to return boundary values.
          grid: a `Grid` object on which to evaluate boundary conditions.
          offset: a Tuple of offsets that specifies (along with grid) where to
            evaluate boundary conditions in space.
          time: a float used as an input to boundary function.

        Returns:
          A tuple of arrays of grid.ndim - 1 dimensions that specify values on the
          boundary. In case of periodic boundaries, returns a tuple(None,None).
        """
        raise NotImplementedError(
            "values() not implemented in BoundaryConditions base class."
        )

    def trim_boundary(
        self,
        u: GridVariable,
    ) -> GridVariable:
        raise NotImplementedError(
            "trim_boundary() not implemented in BoundaryConditions base class."
        )

    def impose_bc(self, u: GridVariable, mode: Optional[str] = "") -> GridVariable:
        """Impose boundary conditions on the grid variable."""
        raise NotImplementedError(
            "impose_bc() not implemented in BoundaryConditions base class."
        )

    def impose_immersed_bc(self, u: GridVariable) -> GridVariable:
        """Impose immersed boundary conditions on the grid variable."""
        raise NotImplementedError(
            "impose_immersed_bc() not implemented in BoundaryConditions base class."
        )

    def pad_and_impose_bc(
        self,
        u: GridVariable,
        offset: Optional[Tuple[float, ...]] = None,
    ) -> GridVariable:
        """Pads the grid variable and imposes boundary conditions."""
        raise NotImplementedError(
            "pad_and_impose_bc() not implemented in BoundaryConditions base class."
        )

    def clone(self, *args, **kwargs) -> BoundaryConditions:
        """Returns a clone of the boundary conditions."""
        raise NotImplementedError(
            "clone() not implemented in BoundaryConditions base class."
        )


def _binary_method(name, op):
    """
    Implement a forward binary method with an operator.
    see np.lib.mixins.NDArrayOperatorsMixin

    Notes: because GridArray is a subclass of torch.Tensor, we need to check
    if the other operand is a GridArray first, otherwise, isinstance(other, _HANDLED_TYPES) will return True as well, which is not what we want as
    there will be no offset in the other operand.
    """

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            elif self.grid != other.grid:
                raise ValueError(
                    f"Cannot operate on arrays with different grids: {self.grid} vs {other.grid}"
                )
            return GridVariable(
                op(self.data, other.data), self.offset, self.grid)
        elif isinstance(other, _HANDLED_TYPES):
            return GridVariable(op(self.data, other), self.offset, self.grid)

        return NotImplemented

    method.__name__ = f"__{name}__"
    return method


def _reflected_binary_method(name, op):
    """Implement a reflected binary method with an operator."""

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            elif self.grid != other.grid:
                raise ValueError(
                    f"Cannot operate on arrays with different grids: {self.grid} vs {other.grid}"
                )
            return GridVariable(op(other.data, self.data), self.offset, self.grid)
        elif isinstance(other, _HANDLED_TYPES):
            return GridVariable(op(other, self.data), self.offset, self.grid)

        return NotImplemented

    method.__name__ = f"__r{name}__"
    return method


def _inplace_binary_method(name, op):
    """
    Implement an in-place binary method with an operator.
    Note: inplace operations do not change the boundary conditions or the offset.
    """

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            elif self.grid != other.grid:
                raise ValueError(
                    f"Cannot operate on arrays with different grids: {self.grid} vs {other.grid}"
                )
            self.data = op(self.data, other.data)
            return self
        elif isinstance(other, _HANDLED_TYPES):
            self.data = op(self.data, other)
            return self

        return NotImplemented

    method.__name__ = f"__i{name}__"
    return method


def _numeric_methods(name, op):
    """Implement forward, reflected and inplace binary methods with an operator."""
    return (
        _binary_method(name, op),
        _reflected_binary_method(name, op),
        _inplace_binary_method(name, op),
    )


def _unary_method(name, op):
    def method(self):
        return GridVariable(op(self.data), self.offset, self.grid, self.bc)

    method.__name__ = f"__i{name}__"
    return method


class GridTensorOpsMixin:
    """
    The implementation refers to that of np.lib.mixins.NDArrayOperatorsMixin
    """

    __slots__ = ()

    __lt__ = _binary_method("lt", operator.lt)
    __le__ = _binary_method("le", operator.le)
    __eq__ = _binary_method("eq", operator.eq)
    __ne__ = _binary_method("ne", operator.ne)
    __gt__ = _binary_method("gt", operator.gt)
    __ge__ = _binary_method("ge", operator.ge)

    __add__, __radd__, __iadd__ = _numeric_methods("add", lambda x, y: x + y)
    __sub__, __rsub__, __isub__ = _numeric_methods("sub", lambda x, y: x - y)
    __mul__, __rmul__, __imul__ = _numeric_methods("mul", lambda x, y: x * y)
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
        "div", lambda x, y: x / y
    )

    # # Unary methods, ~ operator is not implemented
    __neg__ = _unary_method("neg", operator.neg)
    __pos__ = _unary_method("pos", operator.pos)
    __abs__ = _unary_method("abs", operator.abs)


@dataclasses.dataclass
class GridVariable(GridTensorOpsMixin):
    """
    GridVariable class has data with
        - an alignment offset
        - an associated grid
        - a boundary condition

    Offset values in the range [0, 1] fall within a single grid cell.
    Offset enables performing pad and shift operations, e.g. for finite difference calculations, requires boundary condition (BC) information. Since different variables in a PDE system can have different BCs, this class associates a specific variable's data with its BCs.

    Examples:
      offset=(0, 0) means that each point is at the bottom-left corner.
      offset=(0.5, 0.5) is at the grid center.
      offset=(1, 0.5) is centered on the right-side edge.

    Attributes:
      data: torch.Tensor values.
      offset: alignment location of the data with respect to the grid.
      grid: the Grid associated with the array data.
      dtype: type of the array data.
      shape: lengths of the array dimensions.
      bc: boundary conditions for this variable.

    Porting note by S. Cao:
     - [x] defining __init__() or using super().__init__() will cause a recursive loop not sure why (fixed 0.0.6).
     - [x] (added 0.0.1) the original jax implentation uses np.lib.mixins.NDArrayOperatorsMixin
    and the __array_ufunc__ method to implement arithmetic operations
    here it is modified to use torch.Tensor as the base class
    and __torch_function__ to do various things like clone() and to()
    reference: https://pytorch.org/docs/stable/notes/extending.html
     - [x] (added 0.0.8) Mixin defining all operator special methods using __torch_function__. Some integer-based operations are not implemented.
     - [x] (0.1.1) In original Google Research's Jax-CFD code, the devs opted to separate GridArray (no bc) and GridVariable (bc). After carefully studied the FVM implementation, I decided to combine GridArray with GridVariable to reduce code duplication.
     - One can definitely try to use TensorClass from tensordict to implement a more generic GridVariable class, however I found using Tensorclass or @tensorclass actually slows down the code quite a bit.
     - [x] (0.2.0) Finished refactoring the whole GridVariable class for various routines, getting rid of the GridArray class, adding several helper functions for GridVariableVector, and adding batch dimension checks.
     - [x] (0.2.5) Added imposing variable/function-valued nonhomogeneous Dirichlet boundary conditions.
    """

    data: torch.Tensor
    offset: Tuple[float, ...]
    grid: Grid
    bc: Optional[BoundaryConditions] = None

    def __post_init__(self):
        if not isinstance(self.data, torch.Tensor):  # frequently missed by pytype
            raise ValueError(
                f"Expected data type to be torch.Tensor, got {type(self.data)}"
            )
        if self.bc is not None:
            """bc = None follows the original GridArray behavior in Jax-CFD"""
            if len(self.bc.types) != self.grid.ndim:
                raise ValueError(
                    "Incompatible dimension between grid and bc, grid dimension = "
                    f"{self.grid.ndim}, bc dimension = {len(self.bc.types)}"
                )

    def __repr__(self) -> str:
        lines = [f"GridVariable:"]
        display_data = self.disp_data
        lines.append(f"data tensor: \n{display_data.numpy()}\n")
        lines.append(f"data shape: {tuple(s for s in self.data.shape)}")
        lines.append(f"offset: {self.offset}")
        lines.append(f"grid shape: {self.grid.shape}")
        # Add grid domain info
        for i in range(self.grid.ndim):
            lower, upper = self.grid.domain[i]
            step = self.grid.step[i]
            lines.append(f"  dim {i}: [{lower:.3f}, {upper:.3f}], step={step:.3f}")

        lines.append(f"\ndtype : {self.data.dtype}")
        lines.append(f"device: {self.device}")

        # Add boundary condition info if available
        if self.bc is not None:
            bc_repr = repr(self.bc)
            lines.append(f"\nboundary conditions:")
            bc_lines = bc_repr.split("\n")
            for bc_line in bc_lines:
                lines.append(f"  {bc_line}")
        else:
            lines.append(f"boundary conditions: None")

        return "\n".join(lines)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def array(self) -> GridVariable:
        """Returns a GridVariable without boundary conditions
        added for back-compatibility
        The behavior of v.array is similar to Jax-CFD's GridArray
        """
        return GridVariable(self.data, self.offset, self.grid)

    @array.setter
    def array(self, v: GridVariable):
        """Sets data, offset, and grid from another GridVariable, ignoring its boundary conditions."""
        self.data = v.data
        self.offset = v.offset
        self.grid = v.grid
        self.bc = None  # reset boundary conditions

    @property
    def disp_data(self) -> torch.Tensor:
        """Returns the data tensor with the second-to-last dimension flipped. Otherwise return a numpy array for printing."""
        # This is useful for displaying 2D data in a natural way
        disp_data = self.data.clone().cpu().detach()
        if self.grid.ndim >= 2:
            disp_data = torch.flip(disp_data.swapaxes(-2, -1), dims=[-2])
        return disp_data

    @property
    def device(self) -> torch.device:
        return self.data.device

    def norm(self, p: Optional[Union[int, float]] = None, **kwargs) -> torch.Tensor:
        """Returns the norm of the data."""
        return torch.linalg.norm(self.data, p, **kwargs)

    @property
    def L2norm(self) -> torch.Tensor:
        """returns the batched norm, shaped (bsz, )"""
        dims = range(-self.grid.ndim, 0)
        return self.norm(dim=tuple(dims)) * (self.grid.step[0] * self.grid.step[1]) ** (
            1 / self.grid.ndim
        )

    def clone(self):
        return GridVariable(self.data.clone(), self.offset, self.grid, self.bc)

    def to(self, *args, **kwargs):
        return GridVariable(
            self.data.to(*args, **kwargs), self.offset, self.grid, self.bc
        )

    def __getitem__(self, index):
        """Allows indexing into the GridVariable like a tensor."""
        # when slicing only return the tensor data
        new_data = self.data[index]
        return new_data

    def __setitem__(self, index, value):
        """Allows setting items in the GridVariable like a tensor."""
        if isinstance(value, GridVariable):
            self.data[index] = value.data
        else:
            self.data[index] = value

    @staticmethod
    def is_torch_fft_func(func):
        return getattr(func, "__module__", "").startswith("torch._C._fft")

    @staticmethod
    def is_torch_linalg_func(func):
        return getattr(func, "__module__", "").startswith("torch._C._linalg")

    @classmethod
    def __torch_function__(cls, ufunc, types, args=(), kwargs=None):
        """Define arithmetic on GridVariable using an implementation similar to NumPy's NDArrayOperationsMixin."""
        if kwargs is None:
            kwargs = {}
        if not all(issubclass(t, _HANDLED_TYPES + (GridVariable,)) for t in types):
            return NotImplemented
        try:
            # get the corresponding torch function similar to numpy ufunc
            if cls.is_torch_fft_func(ufunc):
                # For FFT functions, we can use the original function
                processed_args = [
                    x.data if isinstance(x, GridVariable) else x for x in args
                ]
                result = ufunc(*processed_args, **kwargs)
                offset = consistent_offset_arrays(
                    *[x for x in args if (type(x) is GridVariable)]
                )
                grid = consistent_grid_arrays(
                    *[x for x in args if isinstance(x, GridVariable)]
                )
                bc = consistent_bc_arrays(
                    *[x for x in args if isinstance(x, GridVariable)]
                )
                return GridVariable(result, offset, grid, bc)
            elif cls.is_torch_linalg_func(ufunc):
                # For linalg functions, we can use the original function
                # acting only on the data
                processed_args = [
                    x.data if isinstance(x, GridVariable) else x for x in args
                ]
                return ufunc(*processed_args, **kwargs)
            else:
                ufunc = getattr(torch, ufunc.__name__)
        except AttributeError as e:
            return NotImplemented

        arrays = [x.data if isinstance(x, GridVariable) else x for x in args]
        result = ufunc(*arrays, **kwargs)

        # If the result is a scalar (0-dim tensor or Python scalar), return it directly
        if isinstance(result, torch.Tensor) and result.ndim == 0:
            return result
        if not isinstance(result, torch.Tensor):
            return result

        offset = consistent_offset_arrays(
            *[x for x in args if isinstance(x, GridVariable)]
        )
        grid = consistent_grid_arrays(*[x for x in args if isinstance(x, GridVariable)])
        # no bc here because functional operations removes the boundary conditions
        if isinstance(result, tuple):
            return tuple(GridVariable(r, offset, grid) for r in result)
        else:
            return GridVariable(result, offset, grid)

    def shift(
        self,
        offset: int,
        dim: int,
    ) -> GridVariable:
        """Shift this GridVariable by `offset`.

        Args:
          offset: positive or negative integer offset to shift.
          dim: axis to shift along.

        Returns:
          A copy of the encapsulated GridVariable, shifted by `offset`. The returned
          GridVariable has offset `u.offset + offset`.
          u.shape is (*, n, m)

        Note:
          The implementation in Jax-CFD does not take into account the batch dimension upon calling u.shift(). This implementation fixed the behavior by using negative indexing.
        """
        return shift(
            self,
            offset=offset,
            dim=dim,
        )

    def trim_boundary(self) -> GridVariable:
        """Returns a GridVariable with boundary conditions trimmed.
        If the BC is None, nothing happens."""
        return self.bc.trim_boundary(self) if self.bc is not None else self

    def _interior_grid(self) -> Grid:
        """Returns only the interior grid points."""
        assert (
            self.bc is not None
        ), "Boundary conditions must be set to get interior grid."
        grid = self.grid
        domain = list(grid.domain)
        shape = list(grid.shape)
        for dim in range(-self.grid.ndim, 0):
            # nothing happens in periodic case
            if self.bc.types[dim][1] == "periodic":
                continue
            # nothing happens if the offset is not 0.0 or 1.0
            # this will automatically set the grid to interior.
            if math.isclose(self.offset[dim], 1.0):
                shape[dim] -= 1
                domain[dim] = (domain[dim][0], domain[dim][1] - grid.step[dim])
            elif math.isclose(self.offset[dim], 0.0):
                shape[dim] -= 1
                domain[dim] = (domain[dim][0] + grid.step[dim], domain[dim][1])
        return Grid(shape, domain=tuple(domain))

    def _interior_array(self) -> torch.Tensor:
        """Returns only the interior points of self.data."""
        assert (
            self.bc is not None
        ), "Boundary conditions must be set to get interior data."
        data = self.data
        for dim in range(self.grid.ndim):
            # nothing happens in periodic case
            if self.bc.types[dim][1] == "periodic":
                continue
            # nothing happens if the offset is not 0.0 or 1.0
            if math.isclose(self.offset[dim], 1.0):
                data, _ = tensor_utils.split_along_axis(data, -1, dim)
            elif math.isclose(self.offset[dim], 0.0):
                _, data = tensor_utils.split_along_axis(data, 1, dim)

        return data

    def interior(self) -> GridVariable:
        """Returns a GridArray associated only with interior points.

         Interior is defined as the following:register
           for d in range(u.grid.ndim):
            points = u.grid.axes(offset=u.offset[d])
            interior_points =
              all points where grid.domain[d][0] < points < grid.domain[d][1]

        The exception is when the boundary conditions are periodic,
        in which case all points are included in the interior.

        In case of dirichlet with edge offset, the grid and array size is reduced,
        since one scalar lies exactly on the boundary. In all other cases,
        self.grid and self.data are returned.

        Porting notes:
        This method actually does not check whether the boundary conditions are imposed or not, is purely determined by the offset.
        """
        interior_array = self._interior_array()
        interior_grid = self._interior_grid()
        return GridVariable(interior_array, self.offset, interior_grid)

    def impose_bc(self, mode: str = "") -> GridVariable:
        """Returns the GridVariable with edge BC enforced, if applicable.

        For GridVariables having nonperiodic BC and offset 0 or 1, there are values
        in the array data that are dependent on the boundary condition.
        impose_bc() changes these boundary values to match the prescribed BC.
        """
        assert self.bc is not None, "Boundary conditions must be set to impose BC."
        return self.bc.impose_bc(self, mode)

    def impose_immersed_bc(self) -> GridVariable:
        """Apply immersed boundary conditions to the GridVariable.

        This method enforces immersed boundary conditions by applying a mask that distinguishes between fluid and solid regions within the computational domain. In solid regions (where the mask is 0.0), the values are set to the prescribed immersed boundary condition value. In fluid regions (where the mask is 1.0), the original values are preserved.

    Returns:
        GridVariable: A new GridVariable with immersed boundary masks applied.

        """
        assert self.bc is not None, "Boundary conditions must be set to impose immersed BC."
        return self.bc.impose_immersed_bc(self)

    def enforce_edge_bc(self, *args) -> GridVariable:
        """Returns the GridVariable with edge BC enforced, if applicable.

        For GridVariables having nonperiodic BC and offset 0 or 1, there are values
        in the array data that are dependent on the boundary condition.
        enforce_edge_bc() changes these boundary values to match the prescribed BC.

        Args:
        *args: any optional values passed into BoundaryConditions values method.
        """
        if self.grid.shape != self.data.shape:
            raise ValueError("Stored array and grid have mismatched sizes.")
        data = torch.as_tensor(self.data).clone()  # Clone to avoid modifying original
        for dim in range(self.grid.ndim):
            if "periodic" not in self.bc.types[dim]:
                values = self.bc.values(dim, self.grid, *args)
                for boundary_side in range(2):
                    if math.isclose(self.offset[dim], boundary_side):
                        # boundary data is set to match self.bc:
                        all_slice = [
                            slice(None, None, None),
                        ] * self.grid.ndim
                        all_slice[dim] = -boundary_side
                        data[tuple(all_slice)] = values[boundary_side]
        return GridVariable(data, self.offset, self.grid, self.bc)




class GridVectorBase(tuple, Generic[T]):
    """
    A tuple-like container for GridVariable objects, representing a vector field.
    Supports elementwise addition and scalar multiplication.
    """

    def __new__(cls, elements: Sequence[T]):
        if not all(isinstance(x, cls._element_type()) for x in elements):
            raise TypeError(
                f"All elements must be instances of {cls._element_type().__name__}. Got {[type(x) for x in elements]}."
            )
        return super().__new__(cls, elements)

    @classmethod
    def _element_type(cls):
        raise NotImplementedError("Subclasses must override _element_type()")

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length.")
        return self.__class__([a + b for a, b in zip(self, other)])

    __radd__ = __add__

    def __iadd__(self, other):
        # Tuples are immutable, so __iadd__ should return a new object using __add__
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length.")
        return self.__class__([a - b for a, b in zip(self, other)])

    __rsub__ = lambda self, other: self.__class__([b - a for a, b in zip(self, other)])

    def __isub__(self, other):
        # Tuples are immutable, so __isub__ should return a new object using __sub__
        return self.__sub__(other)

    def __mul__(self, x):
        if not isinstance(x, _HANDLED_TYPES + (self._element_type(),)):
            return NotImplemented
        return self.__class__([v * x for v in self])

    __imul__ = __rmul__ = __mul__

    def __truediv__(self, x):
        if not isinstance(x, _HANDLED_TYPES):
            return NotImplemented
        return self.__class__([v / x for v in self])

    __itruediv__ = __truediv__

    def __rtruediv__(self, x):
        """
        __rdiv__ does not really make sense for GridVariableVector, but is
        implemented for consistency.
        """
        if not isinstance(x, _HANDLED_TYPES):
            return NotImplemented
        return self.__class__([x / v for v in self])


class GridVariableVector(GridVectorBase[GridVariable]):
    @classmethod
    def _element_type(cls):
        return GridVariable

    def to(self, *args, **kwargs):
        return self.__class__([v.to(*args, **kwargs) for v in self])

    def clone(self, *args, **kwargs):
        return self.__class__([v.clone(*args, **kwargs) for v in self])

    @property
    def shape(self):
        """Return the overall shape of the vector.
        Example:
        v0 = GridVariable(...)
        v1 = GridVariable(...)
        self = GridVariableVector([v0, v1])
        then self.shape will return (2, *shape) where shape = v0.shape.
        """
        if not self:
            return ()
        _shape = self[0].shape
        return torch.Size((len(self),) + _shape)

    @property
    def device(self):
        _device = {v.device for v in self}
        if len(_device) != 1:
            raise ValueError(
                f"All component of {type(self)} must be on the same device."
            )
        return _device.pop()

    @property
    def dtype(self):
        """Return the dtype of the vector."""
        _dtype = {v.dtype for v in self}
        if len(_dtype) != 1:
            raise ValueError(f"All component of {type(self)} must have the same dtype.")
        return _dtype.pop()

    @property
    def data(self) -> Tuple[torch.Tensor, ...]:
        """
        added for compatibility
        """
        return tuple(v.data for v in self)

    @data.setter
    def data(self, values: Tuple[torch.Tensor, ...]):
        for v, value in zip(self, values):
            v.data = value

    @property
    def array(self) -> GridVariableVector:
        """Returns a tuple of GridVariables without boundary conditions."""
        return GridVariableVector(tuple(v.array for v in self))

    @array.setter
    def array(self, values: Tuple[GridVariable, ...]):
        """Sets data, offset, and grid from another GridVariableVector, ignoring its boundary conditions."""
        if len(values) != len(self):
            raise ValueError(
                f"Cannot set array with different length: {len(values)} vs {len(self)}"
            )
        for v, value in zip(self, values):
            v.array = value


class GridTensor(torch.Tensor):
    """An array of GridArrays, representing a physical tensor field.

    Packing tensor coordinates into a torch tensor of dtype object is useful
    because pointwise matrix operations like trace, transpose, and matrix
    multiplications of physical tensor quantities is meaningful.

    TODO:
    Add supports to operations like trace, transpose, and matrix multiplication on physical tensor fields, without register_pytree_node.

    Example usage:
      grad = fd.gradient_tensor(uv)                    # a rank 2 Tensor
      strain_rate = (grad + grad.T) / 2.
      nu_smag = np.sqrt(np.trace(strain_rate.dot(strain_rate)))
      nu_smag = Tensor(nu_smag)                        # a rank 0 Tensor
      subgrid_stress = -2 * nu_smag * strain_rate      # a rank 2 Tensor
    """

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        return torch.Tensor.__new__(cls, data, *args, **kwargs)

    def clone(self, *args, **kwargs):
        return super().clone(*args, **kwargs)


def shift(
    u: GridVariable, offset: int, dim: int, bc: Optional[BoundaryConditions] = None
) -> GridVariable:
    """Shift a GridVariable by `offset`.

    Args:
        u: an `GridVariable` object.
        offset: positive or negative integer offset to shift.
        dim: axis to shift along.

    Returns:
        A copy of `u`, shifted by `offset`. The returned `GridVariable` has offset
        `u.offset + offset`.
    """
    dim = -u.grid.ndim + dim if dim >= 0 else dim
    padded = pad(u, offset, dim, bc)
    trimmed = trim(padded, -offset, dim)
    return trimmed


def pad(
    u: GridVariable,
    width: Union[Tuple[int, int], int],
    dim: int,
    bc: Optional[BoundaryConditions] = None,
    mode: Optional[str] = Padding.EXTEND,
    bc_types: Optional[Tuple[str, str]] = None,
    values: Optional[Union[BCValue, BCValues]] = None,
) -> GridVariable:
    """Pad a GridVariable by `padding`.

    Important: the original _pad in jax_cfd makes no sense past 1 ghost cell for nonperiodic boundaries.
    This is sufficient to replicate jax_cfd finite difference code.

    Args:
        u: an `GridVariable` object.
        width: number of elements to pad along axis. Use negative value for lower
            boundary or positive value for upper boundary.
        dim: axis to pad along.
        bc: boundary conditions to use for padding. If None, uses the boundary conditions of u.
        bc_types: boundary condition types for the dimension `dim`. If None, uses the boundary conditions of u.
        mode: padding mode for Ghost cells! The function tries to automatically select the mode based on the boundary conditions.
        values: tensorial values can be passed directly, see boundaries.DiscreteBoundaryConditions bc_values and pad_and_impose_bc() for details.

    Returns:
        Padded array, elongated along the indicated axis.

    Note:
        - the padding can be only defined when there is proper BC given. If bc is externally given (such as those cases in boundaryies.py), it will override the one in u.bc (if it is not None).
        - In the original Jax-CFD codes, when the width > 1, the code raises error "Padding past 1 ghost cell is not defined in nonperiodic case.", this is now removed and tested for correctness.
    """
    assert not (
        u.bc is None and bc is None and bc_types is None
    ), "u.bc, bc, and bc_types cannot be None at the same time"
    assert mode in [
        Padding.MIRROR,
        Padding.EXTEND,
        Padding.SYMMETRIC,
        Padding.NONE,
    ], f"Padding mode must be one of ['{Padding.MIRROR}', '{Padding.EXTEND}', '{Padding.SYMMETRIC}', None], got '{mode}'"
    bc = bc if bc is not None else u.bc  # use bc in priority
    bc_types = bc.types[dim] if bc_types is None else bc_types
    values = values if values is not None else bc.bc_values
    if isinstance(width, int):
        if width < 0:  # pad lower boundary
            bc_type = bc_types[0]
            padding = (-width, 0)
        else:  # pad upper boundary
            bc_type = bc_types[1]
            padding = (0, width)
    else:
        # when passing width as a tuple
        # the sign choices of the original Jax-CFD is kinda confusing
        # as one is not suppose to pass negative values for lower boundary
        assert width[0] >= 0 and width[1] >= 0, (
            "when passing the padding as a tuple, widths must be non-negative integers, "
            f"got {width} for dim={dim}"
        )
        padding = width

    full_padding = [(0, 0)] * u.grid.ndim
    full_padding[dim] = padding

    new_offset = list(u.offset)
    new_offset[dim] -= padding[0]

    if bc_types[0] != bc_types[1]:
        # a fallback implementation for when the boundary conditions are different
        _bc_types = (bc_types[0], bc_types[0])
        u = pad(u, -padding[0], dim, bc_types=_bc_types, values=values)
        _bc_types = (bc_types[1], bc_types[1])
        u = pad(u, padding[1], dim, bc_types=_bc_types, values=values)
        return u
    else:
        bc_type = bc_types[0]

    if bc_type == BCType.PERIODIC:
        # self.values are ignored here
        data = expand_dims_pad(u.data, full_padding, mode="circular")
        return GridVariable(data, tuple(new_offset), u.grid, bc)
    elif bc_type == BCType.DIRICHLET:
        # adding ghost cells in reality assumes that the padded dims are
        # the SAME on both sides (even though implementation supports and
        # yields correct results for asymmetric padding).
        # for example, if one side is using symmetric padding when computing
        # the ghost cell values, the other side should also use symmetric padding.
        if math.isclose(u.offset[dim] % 1, 0.5):  # cell center
            # make the linearly interpolated value equal to the boundary by setting
            # the padded values to the negative symmetric values
            # on the left side if u.offset is either 0.5 or 1.5, width = -1
            # then (u - bc_val)/u.offset = (u_padded - bc_val)/(u.offset + width)
            # here the implemenation assumes that the left pad and the right pad widths are the same when u.offset == 0.5 and only left needs to be padded when u.offset == 1.5

            if any(p > 1 for p in padding):
                mode = Padding.SYMMETRIC
            _alpha = 1 / u.offset[dim]
            data = _alpha * expand_dims_pad(
                u.data, full_padding, mode="constant", constant_values=values
            ) + (1 - _alpha) * expand_dims_pad(u.data, full_padding, mode=mode)
            return GridVariable(data, tuple(new_offset), u.grid, bc)
        elif math.isclose(u.offset[dim] % 1, 0):  # cell edge
            # First the value on
            # the boundary needs to be added to the array, if not specified by the interior CV values.
            # Then the mirrored ghost cells need to be appended.

            # if only one value is needed, no mode is necessary.
            if (
                math.isclose(sum(full_padding[dim]), 1)
                or math.isclose(sum(full_padding[dim]), 0)
            ) and (0 <= new_offset[dim] <= 1):
                data = expand_dims_pad(
                    u.data, full_padding, mode="constant", constant_values=values
                )
                return GridVariable(data, tuple(new_offset), u.grid, bc)
            elif (
                sum(full_padding[dim]) > 1
                or (new_offset[dim] < 0)
                or (new_offset[dim] > 1)
            ):
                # either (2, 0) or (1, 1) padding in that dimension
                # only triggered when bc.pad_all is called
                if new_offset[dim] < 0 or new_offset[dim] > 1:
                    # if padding beyond the boundary, use the linear extrapolation
                    # if not specified, of new_offset still >=0, use the user define values
                    data = 2 * expand_dims_pad(
                        u.data, full_padding, mode="constant", constant_values=values
                    ) - expand_dims_pad(u.data, full_padding, mode=Padding.MIRROR)
                    return GridVariable(data, tuple(new_offset), u.grid, bc)
                else:
                    if mode == Padding.EXTEND:
                        data = expand_dims_pad(
                            u.data,
                            full_padding,
                            mode="constant",
                            constant_values=values,
                        )
                        return GridVariable(data, tuple(new_offset), u.grid, bc)
                    elif mode == Padding.MIRROR:
                        bc_padding = [(0, 0)] * u.grid.ndim
                        bc_padding[dim] = tuple(1 if pad > 0 else 0 for pad in padding)
                        # subtract the padded cell
                        full_padding_past_bc = [(0, 0)] * u.grid.ndim
                        full_padding_past_bc[dim] = tuple(
                            pad - 1 if pad > 0 else 0 for pad in padding
                        )
                        # here we are adding 0 boundary cell with 0 value
                        expanded_data = expand_dims_pad(
                            u.data, bc_padding, mode="constant", constant_values=(0, 0)
                        )
                        padding_values = list(values)
                        padding_values[dim] = tuple(
                            [p / 2 for p in padding_values[dim]]
                        )
                        data = 2 * expand_dims_pad(
                            u.data,
                            full_padding,
                            mode="constant",
                            constant_values=padding_values,
                        ) - expand_dims_pad(
                            expanded_data, full_padding_past_bc, mode=mode
                        )
                        return GridVariable(data, tuple(new_offset), u.grid, bc)
                    elif mode == Padding.SYMMETRIC:
                        # symmetric padding, mirrors values at the boundaries
                        data = 2 * expand_dims_pad(
                            u.data,
                            full_padding,
                            mode="constant",
                            constant_values=values,
                        ) - expand_dims_pad(u.data, full_padding, mode=mode)
                        return GridVariable(data, tuple(new_offset), u.grid, bc)
                    else:
                        raise ValueError(
                            f"Unsupported padding mode '{mode}' for Dirichlet BC with cell edge offset"
                        )
            else:
                raise ValueError(
                    f"invalid padding width for Dirichlet BC, expected padding[dim={dim}] to have sum >= 0, got {padding[dim]}"
                )

        else:
            raise ValueError(
                "expected the new offset to be an edge or cell center, got "
                f"offset[dim={dim}]={u.offset[dim]}"
            )
    elif bc_type == BCType.NEUMANN:
        if not (
            math.isclose(u.offset[dim] % 1, 0) or math.isclose(u.offset[dim] % 1, 0.5)
        ):
            raise ValueError(
                "expected offset to be an edge or cell center, got "
                f"offset[dim={dim}]={u.offset[dim]}"
            )
        else:
            # When the data is cell-centered, computes the backward difference.
            # When the data is on cell edges, boundary is set such that
            # (u_boundary - u_last_interior)/grid_step = neumann_bc_value (fixed from Jax-cfd).
            # note: Jax-cfd implementation was wrong, Neumann BC is \nabla u \cdot exterior normal, the order is reversed
            data = expand_dims_pad(
                u.data, full_padding, mode="replicate"
            ) + u.grid.step[dim] * (
                expand_dims_pad(
                    u.data,
                    full_padding,
                    mode="constant",
                    constant_values=values,
                )
                - expand_dims_pad(u.data, full_padding, mode="constant")
            )
            return GridVariable(data, tuple(new_offset), u.grid, bc)

    else:
        raise ValueError("invalid boundary type")


def trim(u: GridVariable, width: int, dim: int) -> GridVariable:
    """Trim padding from a GridVariable.

    Args:
      u: data.
      width: number of elements to trim along axis. Use negative value for lower
        boundary or positive value for upper boundary.
      dim: axis to trim along.

    Returns:
      Trimmed array, shrunk along the indicated axis.
    """
    if width < 0:  # trim lower boundary
        padding = (-width, 0)
    else:  # trim upper boundary
        padding = (0, width)

    limit_index = u.shape[dim] - padding[1]
    data = u.data.index_select(
        dim=dim, index=torch.arange(padding[0], limit_index, device=u.device)
    )
    new_offset = list(u.offset)
    new_offset[dim] += padding[0]
    return GridVariable(data, tuple(new_offset), u.grid)


def expand_dims_pad(
    inputs: torch.Tensor,
    pad: Sequence[Tuple[int, int]],
    mode: str = "constant",
    constant_values: Union[float, BCValues] = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Porting notes:
    - This function serves as replacement for jax.numpy.pad in the original Jax-CFD implementation, basically a wrapper for F.pad with a dimension checker
    - jnp's pad pad_width starts from the first dimension to the last dimension
    while torch's pad pad_width starts from the last dimension to the previous dimension
    example:
    - for torch.nn.functional.pad: pad = (1, 1, 2, 2) means padding last dim by (1, 1) and 2nd to last by (2, 2), the pad arg for the expand_dims_pad function should be ((2, 2), (1, 1)) which the natural ordering of dimensions

    Args:
      inputs: torch.Tensor or a tuple of arrays to pad.
      pad_width: padding width for each dimension.
      mode: padding mode, one of 'constant', 'reflect', 'symmetric'.
      values: values to pad with.

    Returns:
      Padded `inputs`.
    """
    if len(pad) == inputs.ndim:
        # if pad has the same length as inputs.ndim, add a batch dimension
        unsqueezed = True
        inputs = inputs.unsqueeze(0)
    elif len(pad) <= inputs.ndim - 1:
        # if pad has the same length as inputs.ndim - 1, no need to unsqueeze
        unsqueezed = False
    else:
        raise ValueError(f"pad length must <= {inputs.ndim}, got {len(pad)}")

    flat_pad = reduce(lambda a, x: x + a, pad, ())  # flatten the pad and reverse
    if mode == "constant":
        if isinstance(constant_values, (int, float)):
            array = F.pad(inputs, flat_pad, mode=mode, value=constant_values)
        elif isinstance(constant_values, (tuple, list)):
            array = _constant_pad_tensor(inputs, pad, constant_values)
        else:
            raise NotImplementedError(
                f"constant_values must be a float or a tuple/list, got {type(constant_values)}"
            )
    elif mode in ["circular", "reflect", "replicate"]:
        # periodic boundary condition
        array = F.pad(inputs, flat_pad, mode=mode)
    elif mode == "symmetric":
        # symmetric padding, mirrors values at the boundaries
        array = _symmetric_pad_tensor(inputs, pad)
    else:
        raise NotImplementedError(f"invalid mode {mode} for torch.nn.functional.pad")

    return array.squeeze(0) if unsqueezed else array


def _constant_pad_tensor(
    inputs: torch.Tensor,
    pad: Sequence[Tuple[int, int]],
    constant_values: Sequence[Tuple[BCValue, BCValue]],
    **kwargs,
) -> torch.Tensor:
    """
    Corrected padding function that supports different constant/tensor values for each side.
    Pads each dimension from first to last as per the user input, bypassing PyTorch's F.pad behavior of last-to-first padding order.

    Extended to support tensor values for padding - if the input is already a correctly shaped tensor,
    it will be used directly for concatenation instead of creating a new tensor with torch.full.

    Args:
        inputs: torch.Tensor to pad.
        pad: padding width for each dimension, e.g. ((2, 2), (1, 1)) for 2D tensor. (2, 2) means padding the first dimension by 2 on both sides, and (1, 1) means padding the second dimension by 1 on both sides.
        constant_values: constant values to pad with for each dimension, e.g. ((0, 0), (1, 1)). Can also be tensors with correct boundary shapes.

    Example:
    If pad = ((2, 2), (1, 1)) is given for a 2D (potentially batched) tensor of shape (*, 10, 20),
        - pad[1] corresponds to the padding of the last dimension (20),
        - pad[0] corresponds to the padding of the second-to-last dimension (10).
        - the resulting tensor shape will be (*, 10 + 2 + 2, 20 + 1 + 1) = (*, 14, 22)

        >>> data = torch.tensor([[11., 12., 13., 14.]])
        >>> _constant_pad_tensor(data, ((2, 1),), ((0, 1),))
        tensor([[0., 0., 11., 12., 13., 14., 1.],])

    """
    # inputs was unsqueezed at dim 0, so actual data dims are shifted by +1
    dims_to_pad = len(pad)  # number of dimensions to pad
    result = inputs

    for i in reversed(range(dims_to_pad)):  # iterate from last to first dimension
        dim = i - dims_to_pad
        # correct mapping from pad index to tensor dim

        left_pad, right_pad = pad[i]

        if left_pad == 0 and right_pad == 0:
            continue

        # Get constant values
        if len(constant_values) > 0:
            if (
                isinstance(constant_values[i], (tuple, list))
                and len(constant_values[i]) == 2
            ):
                left_val, right_val = constant_values[i]
            else:
                left_val = right_val = constant_values[i]
        else:
            left_val = right_val = 0.0

        # Handle left padding
        if left_pad > 0:
            left_tensor = (
                _create_boundary_tensor(left_val, result.shape, dim, left_pad)
                .to(result.dtype)
                .to(result.device)
            )
            result = torch.cat([left_tensor, result], dim=dim)

        # Handle right padding
        if right_pad > 0:
            right_tensor = (
                _create_boundary_tensor(right_val, result.shape, dim, right_pad)
                .to(result.dtype)
                .to(result.device)
            )
            result = torch.cat([result, right_tensor], dim=dim)

    return result


def _create_boundary_tensor(
    value: Union[BCValue, float],
    target_shape: Tuple[int, ...],
    pad_dim: int,
    pad_width: int,
) -> torch.Tensor:
    """
    Create a boundary tensor for padding with proper shape handling.

    Args:
        value: The boundary value (tensor, scalar, or None)
        target_shape: Shape of the tensor being padded
        pad_dim: The dimension being padded (negative indexing)
        pad_width: Width of padding for this side

    Returns:
        Properly shaped tensor for concatenation

    Notes:
        current only handle 1D bc for concatenation with
        (b, nx, ny) shaped tensors.
    """
    # Calculate expected boundary shape
    expected_shape = list(target_shape)
    expected_shape[pad_dim] = pad_width

    if isinstance(value, torch.Tensor):
        # Handle tensor boundary values
        if list(value.shape) == expected_shape:
            # Tensor already has correct shape
            return value

        elif value.ndim == 1:
            # Handle 1D boundary tensor - need to properly reshape/expand
            if value.shape[0] > 1:
                boundary_size = value.shape[0]

                # Determine target size for the boundary dimension
                if pad_dim == -1:  # Last dimension
                    # For padding last dim, boundary values correspond to second-to-last dim
                    boundary_dim = -2
                    target_size = expected_shape[boundary_dim]
                elif pad_dim == -2:  # Second to last dimension
                    # For padding second-to-last dim, boundary values correspond to last dim
                    boundary_dim = -1
                    target_size = expected_shape[boundary_dim]
                else:
                    raise NotImplementedError(
                        f"Padding BC for dimension {pad_dim} is not implemented. "
                        "Currently only -1 (y dimension) and -2 (x dimension) are supported."
                    )

                # Handle size mismatch with interpolation
                if boundary_size != target_size:
                    # Use interpolation to resize the boundary tensor
                    # Reshape to [1, 1, boundary_size] for F.interpolate
                    # TODO: interpolate here is a janky monkey patch
                    # as the location may be mis-aligned
                    # a better way
                    # TODO: a better way to handle this is simply use slicing

                    reshaped_for_interp = value[None, None, :]
                    interpolated = F.interpolate(
                        reshaped_for_interp,
                        size=target_size,
                        mode="linear",
                        align_corners=False,
                    )
                    # Remove the added dimensions: [1, 1, target_size] -> [target_size]
                    value = interpolated.squeeze()
                    boundary_size = target_size

                # Create the target tensor by using view and expand appropriately
                new_shape = [1] * len(expected_shape)
                new_shape[boundary_dim] = boundary_size

                # Reshape and expand
                reshaped = value.view(new_shape)
                boundary_tensor = reshaped.expand(expected_shape)

                return boundary_tensor
            elif value.shape[0] == 1:
                # If the tensor has only one element, we can simply expand it
                # to the expected shape
                return value.expand(expected_shape)
            else:
                raise ValueError(
                    f"Boundary tensor with {value.ndim}D shape {value.shape} is not supported. "
                    f"Only 1D boundary tensors are supported."
                )

        else:
            raise ValueError(
                f"Boundary tensor with {value.ndim}D shape {value.shape} is not supported. "
                f"Only 1D boundary tensors are supported."
            )

    else:
        # Handle scalar boundary values
        scalar_val = float(value) if value is not None else 0.0
        return torch.full(expected_shape, scalar_val)


def _symmetric_pad_tensor(
    inputs: torch.Tensor,
    pad: Sequence[Tuple[int, int]],
    **kwargs,
) -> torch.Tensor:
    """
    Symmetric padding function that mirrors values at the boundaries.
    Pads each dimension from first to last as per the user input.
    This is a drop-in replacement for np.pad with mode == 'symmetric'.

    Args:
        inputs: torch.Tensor to pad.
        pad: padding width for each dimension, e.g. ((2, 2), (1, 1)) for 2D tensor.
             (2, 2) means padding the first dimension by 2 on both sides,
             and (1, 1) means padding the second dimension by 1 on both sides.

    Example:
    If ((2, 2), (1, 1)) is given for a 2D (potentially batched) tensor of shape (*, 10, 20),
        - pad[1] corresponds to the padding of the last dimension (20),
        - pad[0] corresponds to the padding of the second-to-last dimension (10).
        - the resulting tensor shape will be (*, 10 + 2 + 2, 20 + 1 + 1) = (*, 14, 22)

        >>> data = torch.tensor([[11., 12., 13., 14.]])
        >>> _symmetric_pad_tensor(data, ((2, 0),))
        tensor([[12., 11., 11., 12., 13., 14.],])

    Note:
        - the 'reflect' mode of F.pad would yield for the data above
        tensor([[13., 12., 11., 12., 13., 14.],])

    """
    dims_to_pad = len(pad)  # number of dimensions to pad
    result = inputs

    for i in reversed(range(dims_to_pad)):
        dim = i - dims_to_pad
        # correct mapping from pad index to tensor dim

        left_pad, right_pad = pad[i]

        if left_pad == 0 and right_pad == 0:
            continue

        n = result.shape[dim]
        assert (
            left_pad <= n and right_pad <= n
        ), f"padding must be <= than existing size in dimension (got {left_pad},{right_pad} for size {n})"

        # Get left padding values (mirror from beginning)
        if left_pad > 0:
            # flip the first left_pad elements in the to-be-padded dimension
            left_tensor = result.narrow(dim, 0, left_pad).flip(dims=[dim])
            result = torch.cat([left_tensor, result], dim=dim)

        # Get right padding values (mirror from end)
        if right_pad > 0:
            # flip the last right_pad elements in the to-be-padded dimension
            right_tensor = result.narrow(
                dim, result.shape[dim] - right_pad, right_pad
            ).flip(dims=[dim])
            result = torch.cat([result, right_tensor], dim=dim)

    return result


def averaged_offset(*offsets: List[Tuple[float, ...]]) -> Tuple[float, ...]:
    """Returns the averaged offset of the given arrays."""
    n = len(offsets)
    assert n > 0, "No offsets provided"
    m = len(offsets[0])
    return tuple(sum([o[i] for o in offsets]) / n for i in range(m))


def averaged_offset_arrays(*arrays: GridVariable) -> Tuple[float, ...]:
    """Returns the averaged offset of the given arrays."""
    offsets = list([array.offset for array in arrays])
    return averaged_offset(*offsets)


def control_volume_offsets(*offsets) -> Tuple[Tuple[float, ...], ...]:
    """Returns offsets for the faces of the control volume centered at `offsets`."""
    return tuple(
        tuple(o + 0.5 if i == j else o for i, o in enumerate(offsets))
        for j in range(len(offsets))
    )


def control_volume_offsets_arrays(
    c: GridVariable,
) -> Tuple[Tuple[float, ...], ...]:
    """Returns offsets for the faces of the control volume centered at `c`."""
    return control_volume_offsets(*c.offset)


def consistent_offset_arrays(*arrays: GridVariable) -> Tuple[float, ...]:
    """Returns the unique offset, or raises InconsistentOffsetError."""
    offsets = {array.offset for array in arrays}
    if len(offsets) != 1:
        raise Exception(f"arrays do not have a unique offset: {offsets}")
    (offset,) = offsets
    return offset


def consistent_grid_arrays(*arrays: GridVariable):
    """Returns the unique grid, or raises InconsistentGridError."""
    grids = {array.grid for array in arrays}
    if len(grids) != 1:
        raise Exception(f"arrays do not have a unique grid: {grids}")
    (grid,) = grids
    return grid


def consistent_bc_arrays(*arrays: GridVariable):
    """Returns the unique bc, or return None if difference BC."""
    bcs = {array.bc for array in arrays}
    if len(bcs) != 1:
        return None
    (bc,) = bcs
    return bc


def consistent_grid(grid: Grid, *arrays: GridVariable):
    """Returns the unique grid, or raises InconsistentGridError."""
    grids = {array.grid for array in arrays}
    if len(grids.union({grid})) != 1:
        raise Exception(
            f"arrays' grids {grids} are not consistent with the grid {grid}"
        )
    (grid,) = grids
    return grid


def domain_interior_masks(grid: Grid):
    """Returns cell face arrays with 1 on the interior, 0 on the boundary."""
    masks = []
    for offset in grid.cell_faces:
        mesh = grid.mesh(offset)
        mask = torch.ones(mesh[0].shape, device=grid.device)
        for i, x in enumerate(mesh):
            lower = (
                ~torch.isclose(x, torch.tensor(grid.domain[i][0], device=grid.device))
            ).int()
            upper = (
                ~torch.isclose(x, torch.tensor(grid.domain[i][1], device=grid.device))
            ).int()
            mask = mask * upper * lower
        masks.append(mask)
    return tuple(masks)


def repeat(x: GridVariable, pattern: str, **kwargs) -> GridVariable:
    data = _repeat(x.data, pattern, **kwargs)
    return GridVariable(data, x.offset, x.grid, x.bc)


def rearrange(x: GridVariable, pattern: str, **kwargs) -> GridVariable:
    data = _rearrange(x.data, pattern, **kwargs)
    return GridVariable(data, x.offset, x.grid, x.bc)


def stack_gridvariables(*arrays: GridVariable, dim: int = 0) -> GridVariable:
    """Stack GridVariable objects by stacking their data and preserving metadata."""

    # Extract data from all tensors
    data = [t.data for t in arrays]
    stacked_data = torch.stack(data, dim=dim)

    # Use the first tensor's metadata (assuming they're compatible)
    return GridVariable(
        stacked_data,
        consistent_offset_arrays(*arrays),
        consistent_grid_arrays(*arrays),
        consistent_bc_arrays(*arrays) if arrays[0].bc is not None else None,
    )


def stack_gridvariable_vectors(
    *vectors: GridVariableVector, dim: int = 0
) -> GridVariableVector:
    """Stack GridVariableVector objects (vector fields) by stacking their components and preserving metadata.

    Args:
        *vectors: GridVariableVector to stack
        dim: dimension along which to stack (default: 0)

    Returns:
        GridVariableVector with stacked components

    Example:
        v1 = GridVariableVector([u1, v1])  # velocity field 1
        v2 = GridVariableVector([u2, v2])  # velocity field 2
        stacked = stack_gridvariable_vectors(v1, v2, dim=0)
        # stacked[0].shape: (2, ...)
    """

    vector_length = len(vectors[0])
    if not all(len(v) == vector_length for v in vectors):
        raise ValueError("All vectors must have the same number of components")

    stacked_vector = []
    for i in range(vector_length):
        cs = [v[i] for v in vectors]
        # Stack these components
        c = stack_gridvariables(*cs, dim=dim)
        stacked_vector.append(c)

    return GridVariableVector(stacked_vector)
