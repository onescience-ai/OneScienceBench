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
# ported Google's Jax-CFD functional template to PyTorch's tensor ops

from __future__ import annotations

from collections.abc import Sequence

from typing import Any, Tuple, Union

import torch


def _normalize_axis(axis: int, ndim: int) -> int:
    """Validates and returns positive `axis` value."""
    if not -ndim <= axis < ndim:
        raise ValueError(f"invalid axis {axis} for ndim {ndim}")
    if axis < 0:
        axis += ndim
    return axis


def slice_along_axis(
    inputs: Any, axis: int, idx: Union[slice, int], expect_same_dims: bool = True
) -> Any:
    """Returns slice of `inputs` defined by `idx` along axis `axis`.

    Args:
      inputs: torch.Tensor or a tuple/list of arrays to slice.
      axis: axis along which to slice the `inputs`.
      idx: index or slice along axis `axis` that is returned.
      expect_same_dims: whether all arrays should have same number of dimensions.

    Returns:
      Slice of `inputs` defined by `idx` along axis `axis`.
    """
    if isinstance(inputs, torch.Tensor):
        # Single tensor case
        ndim = inputs.ndim
        slc = tuple(
            idx if j == _normalize_axis(axis, ndim) else slice(None)
            for j in range(ndim)
        )
        return inputs[slc]

    elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        # Tuple/list of tensors case
        arrays = inputs
        ndims = set(a.ndim for a in arrays if isinstance(a, torch.Tensor))
        if expect_same_dims and len(ndims) != 1:
            raise ValueError(
                "arrays in `inputs` expected to have same ndims, but have "
                f"{ndims}. To allow this, pass expect_same_dims=False"
            )

        sliced = []
        for array in arrays:
            if isinstance(array, torch.Tensor):
                ndim = array.ndim
                slc = tuple(
                    idx if j == _normalize_axis(axis, ndim) else slice(None)
                    for j in range(ndim)
                )
                sliced.append(array[slc])
            else:
                # Non-tensor items pass through unchanged
                sliced.append(array)

        # Return same type as input
        return type(inputs)(sliced)

    else:
        # Fallback: return as-is for unsupported types
        return inputs


def split_along_axis(
    inputs: Any, split_idx: int, axis: int, expect_same_dims: bool = True
) -> Tuple[Any, Any]:
    """Returns a tuple of slices of `inputs` split along `axis` at `split_idx`.

    Args:
      inputs: sequence of arrays to split.
      split_idx: index along `axis` where the second split starts.
      axis: axis along which to split the `inputs`.
      expect_same_dims: whether all arrays should have same number of dimensions.

    Returns:
      Tuple of slices of `inputs` split along `axis` at `split_idx`.
    """

    first_slice = slice_along_axis(inputs, axis, slice(0, split_idx), expect_same_dims)
    second_slice = slice_along_axis(
        inputs, axis, slice(split_idx, None), expect_same_dims
    )
    return first_slice, second_slice


def split_axis(inputs: Any, dim: int, keep_dims: bool = False) -> Tuple[Any, ...]:
    """Splits the arrays in `inputs` along `axis`.

    Args:
      inputs: tensor or sequence of tensors to be split.
      dim: axis along which to split the `inputs`.
      keep_dims: whether to keep `dim` dimension.

    Returns:
      Tuple of tensors/sequences that correspond to slices of `inputs` along `dim`. The
      `dim` dimension is removed if `keep_dims` is set to False.

    Raises:
      ValueError: if arrays in `inputs` don't have unique size along `dim`.
    """
    if isinstance(inputs, torch.Tensor):
        # Single tensor case
        dim = _normalize_axis(dim, inputs.ndim)
        axis_size = inputs.shape[dim]

        # Split into individual slices
        splits = []
        for i in range(axis_size):
            slc = tuple(i if j == dim else slice(None) for j in range(inputs.ndim))
            split_tensor = inputs[slc]
            if not keep_dims:
                split_tensor = torch.squeeze(split_tensor, dim)
            splits.append(split_tensor)

        return tuple(splits)

    elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        # Sequence of tensors case
        arrays = [a for a in inputs if isinstance(a, torch.Tensor)]

        if not arrays:
            # No tensors to split, return inputs as-is
            return (inputs,)

        # Check that all tensors have the same size along the split dimension
        axis_shapes = set(a.shape[dim] for a in arrays)
        if len(axis_shapes) != 1:
            raise ValueError(f"Arrays must have equal sized axis but got {axis_shapes}")

        (axis_size,) = axis_shapes

        # Split each tensor and collect results
        splits = []
        for i in range(axis_size):
            split_items = []
            for item in inputs:
                if isinstance(item, torch.Tensor):
                    dim_norm = _normalize_axis(dim, item.ndim)
                    slc = tuple(
                        i if j == dim_norm else slice(None) for j in range(item.ndim)
                    )
                    split_tensor = item[slc]
                    if not keep_dims:
                        split_tensor = torch.squeeze(split_tensor, dim_norm)
                    split_items.append(split_tensor)
                else:
                    # Non-tensor items pass through unchanged
                    split_items.append(item)

            # Return same type as input
            splits.append(type(inputs)(split_items))

        return tuple(splits)

    else:
        # Fallback: return as-is for unsupported types
        return (inputs,)