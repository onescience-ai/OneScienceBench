"""GenCast global-forecast regional query adapter.

The adapter intentionally sits after the official global GenCast predictor.
It consumes an xarray-compatible Dataset whose data variables have the
official ensemble/time/latitude/longitude dimensions and returns a new
Dataset with the same variables and non-spatial metadata.
"""

from dataclasses import dataclass
from copy import deepcopy
import inspect
from typing import Any, Mapping, Optional, Sequence

import numpy as np


_DIMS = ("sample", "time", "lat", "lon")


@dataclass(frozen=True)
class RegionSpec:
    """Inclusive latitude/longitude bounds in degrees."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RegionSpec":
        if not isinstance(value, Mapping):
            raise TypeError("region must be a mapping or RegionSpec")
        required = ("lat_min", "lat_max", "lon_min", "lon_max")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"region is missing keys: {missing}")
        bounds = tuple(float(value[key]) for key in required)
        if not np.all(np.isfinite(bounds)):
            raise ValueError("region bounds must be finite")
        if not (-90.0 <= bounds[0] <= 90.0 and -90.0 <= bounds[1] <= 90.0):
            raise ValueError("latitude bounds must be within [-90, 90]")
        return cls(*bounds)


def _as_region(region: Optional[Mapping[str, Any] | RegionSpec]) -> Optional[RegionSpec]:
    if region is None:
        return None
    return region if isinstance(region, RegionSpec) else RegionSpec.from_mapping(region)


def _axis(values: Any, name: str) -> np.ndarray:
    array = np.asarray(getattr(values, "values", values), dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty one-dimensional finite array")
    return array.copy()


def _normalise_longitudes(values: Any) -> np.ndarray:
    result = np.mod(_axis(values, "longitudes"), 360.0)
    if np.unique(np.round(result, decimals=10)).size != result.size:
        raise ValueError("longitudes must be unique after normalization")
    return result


def _strict_latitudes(values: Any) -> np.ndarray:
    result = _axis(values, "latitudes")
    differences = np.diff(result)
    if differences.size and not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("latitudes must be strictly monotonic")
    return result


def _forecast_times(values: Any) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if not np.issubdtype(result.dtype, np.datetime64):
        raise TypeError("times must contain datetimes")
    if np.any(np.isnat(result)):
        raise ValueError("times must not contain NaT")
    if result.size > 1 and np.any(np.diff(result) <= np.timedelta64(0, "ns")):
        raise ValueError("times must be strictly increasing")
    return result.copy()


def _coord_values(dataset: Any, name: str) -> np.ndarray:
    coords = getattr(dataset, "coords", None)
    if coords is None or name not in coords:
        raise ValueError(f"GenCast Dataset must provide coordinate {name!r}")
    return np.asarray(getattr(coords[name], "values", coords[name]))


def _data_vars(dataset: Any) -> Mapping[str, Any]:
    data_vars = getattr(dataset, "data_vars", None)
    if data_vars is None or not hasattr(data_vars, "items"):
        raise TypeError("predictions must be an xarray-compatible Dataset")
    if not data_vars:
        raise ValueError("GenCast Dataset must contain at least one data variable")
    return data_vars


def _array_values(data_array: Any, name: str) -> np.ndarray:
    dims = tuple(getattr(data_array, "dims", ()))
    if dims != _DIMS:
        raise ValueError(f"variable {name!r} must use dimensions {_DIMS}, got {dims}")
    values = np.asarray(getattr(data_array, "values", data_array))
    if values.ndim != 4 or not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"variable {name!r} must be a numeric 4-D array")
    return values


def _indices(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    region: RegionSpec,
    allow_empty: bool = False,
):
    lat_lo, lat_hi = sorted((region.lat_min, region.lat_max))
    lat_indices = np.flatnonzero(
        (latitudes >= lat_lo - 1e-9) & (latitudes <= lat_hi + 1e-9)
    )
    if lat_indices.size == 0 and not allow_empty:
        raise ValueError("region selects no latitude grid points")

    wrapped = _normalise_longitudes(longitudes)
    start, end = np.mod(region.lon_min, 360.0), np.mod(region.lon_max, 360.0)
    if abs(region.lon_max - region.lon_min) >= 360.0 - 1e-9:
        lon_indices = np.arange(wrapped.size, dtype=np.int64)
    elif -180.0 <= region.lon_min < 180.0 and -180.0 <= region.lon_max < 180.0:
        signed = np.where(wrapped >= 180.0, wrapped - 360.0, wrapped)
        if region.lon_min <= region.lon_max:
            chosen = np.flatnonzero(
                (signed >= region.lon_min - 1e-9)
                & (signed <= region.lon_max + 1e-9)
            )
            lon_indices = chosen[np.argsort(signed[chosen], kind="stable")]
        else:
            unwrapped = np.where(signed < region.lon_min, signed + 360.0, signed)
            chosen = np.flatnonzero(
                (unwrapped >= region.lon_min - 1e-9)
                & (unwrapped <= region.lon_max + 360.0 + 1e-9)
            )
            lon_indices = chosen[np.argsort(unwrapped[chosen], kind="stable")]
    elif start <= end:
        chosen = np.flatnonzero(
            (wrapped >= start - 1e-9) & (wrapped <= end + 1e-9)
        )
        lon_indices = chosen[np.argsort(wrapped[chosen], kind="stable")]
    else:
        east = np.flatnonzero(wrapped >= start - 1e-9)
        west = np.flatnonzero(wrapped <= end + 1e-9)
        east = east[np.argsort(wrapped[east], kind="stable")]
        west = west[np.argsort(wrapped[west], kind="stable")]
        lon_indices = np.concatenate((east, west))
    if lon_indices.size == 0 and not allow_empty:
        raise ValueError("region selects no longitude grid points")
    return lat_indices.astype(np.int64), lon_indices.astype(np.int64)


def _interp_axis(values: np.ndarray, source: np.ndarray, target: np.ndarray, axis: int) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.size == 0:
        raise ValueError("cannot interpolate an empty source axis")
    differences = np.diff(source)
    if differences.size and np.all(differences < 0):
        source = source[::-1]
        values = np.flip(values, axis=axis)
    elif differences.size and not np.all(differences > 0):
        raise ValueError("source grid must be strictly monotonic for interpolation")
    if np.any(target < source[0] - 1e-7) or np.any(target > source[-1] + 1e-7):
        raise ValueError("target grid lies outside the selected region")
    moved = np.moveaxis(values, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    result = np.stack([np.interp(target, source, row) for row in flat], axis=0)
    result = result.reshape(moved.shape[:-1] + (target.size,))
    return np.moveaxis(result, -1, axis)


def _longitude_interpolation_axis(source: np.ndarray, indices: np.ndarray, region: RegionSpec) -> np.ndarray:
    """Return a monotonic eastward axis for the selected longitude points."""
    selected = np.mod(source[indices], 360.0)
    start = np.mod(region.lon_min, 360.0)
    return start + np.mod(selected - start, 360.0)


def _target_axis(values: Sequence[float], name: str, wrap_longitude: bool = False) -> np.ndarray:
    result = _axis(values, name)
    if np.unique(np.round(result, decimals=10)).size != result.size:
        raise ValueError(f"{name} must not contain duplicate coordinates")
    if wrap_longitude and np.unique(np.round(np.mod(result, 360.0), decimals=10)).size != result.size:
        raise ValueError(f"{name} must not contain duplicate coordinates modulo 360")
    return result


def _region_span(region: RegionSpec) -> float:
    if abs(region.lon_max - region.lon_min) >= 360.0 - 1e-9:
        return 360.0
    start, end = region.lon_min, region.lon_max
    return end - start if end >= start else end + 360.0 - start


def _make_dataset(
    data_vars: Mapping[str, np.ndarray],
    coords: Mapping[str, Any],
    attrs: Mapping[str, Any],
    variable_attrs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    coord_attrs: Optional[Mapping[str, Mapping[str, Any]]] = None,
):
    """Construct a native xarray Dataset when available, else a tiny duck-typed object."""
    try:
        import xarray as xr

        xr_coords = {}
        for name, value in coords.items():
            if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], (tuple, list)):
                dims, values = tuple(value[0]), value[1]
                coordinate_attributes = value[2] if len(value) > 2 else {}
            else:
                dims, values = (name,), value
                coordinate_attributes = (coord_attrs or {}).get(name, {})
            xr_coords[name] = xr.DataArray(
                values,
                dims=dims,
                attrs=deepcopy(dict(coordinate_attributes or {})),
                name=name,
            )
        return xr.Dataset(
            data_vars={
                name: (_DIMS, values, deepcopy((variable_attrs or {}).get(name, {})))
                for name, values in data_vars.items()
            },
            coords=xr_coords,
            attrs=deepcopy(dict(attrs)),
        )
    except ImportError:
        return _DatasetFallback(data_vars, coords, attrs, variable_attrs, coord_attrs)


class _ArrayFallback:
    def __init__(self, values: Any, dims: Sequence[str], attrs: Optional[Mapping[str, Any]] = None):
        self.values = np.asarray(values)
        self.dims = tuple(dims)
        self.attrs = dict(attrs or {})


class _DatasetFallback:
    def __init__(
        self,
        data_vars: Mapping[str, Any],
        coords: Mapping[str, Any],
        attrs: Mapping[str, Any],
        variable_attrs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        coord_attrs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        self.data_vars = {
            name: value
            if hasattr(value, "dims")
            else _ArrayFallback(value, _DIMS, deepcopy((variable_attrs or {}).get(name)))
            for name, value in data_vars.items()
        }
        self.coords = {}
        for name, value in coords.items():
            if hasattr(value, "values"):
                self.coords[name] = value
            elif isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], (tuple, list)):
                dims, values = tuple(value[0]), value[1]
                coord_spec_attrs = value[2] if len(value) > 2 else {}
                self.coords[name] = _ArrayFallback(values, dims, deepcopy(coord_spec_attrs))
            else:
                self.coords[name] = _ArrayFallback(
                    value, (name,), deepcopy((coord_attrs or {}).get(name))
                )
        self.attrs = deepcopy(dict(attrs))


class GenCastRegionalInference:
    """Query a GenCast global ensemble Dataset on a geographic sub-grid."""

    def __init__(
        self,
        latitudes: Sequence[float],
        longitudes: Sequence[float],
        variables: Optional[Sequence[str]] = None,
        times: Optional[Sequence[Any]] = None,
        predictor: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        if predictor is not None and model is not None:
            raise ValueError("provide either predictor or model, not both")
        self.latitudes = _strict_latitudes(latitudes)
        self.longitudes = _axis(longitudes, "longitudes")
        _normalise_longitudes(self.longitudes)
        self.variables = None if variables is None else tuple(str(value) for value in variables)
        self.times = None if times is None else tuple(_forecast_times(times))
        self.predictor = predictor if predictor is not None else model
        self.model = self.predictor

    def _validate_dataset(self, dataset: Any):
        variables = _data_vars(dataset)
        sample = _coord_values(dataset, "sample")
        times = _coord_values(dataset, "time")
        latitudes = _axis(_coord_values(dataset, "lat"), "lat coordinate")
        longitudes = _axis(_coord_values(dataset, "lon"), "lon coordinate")
        if sample.ndim != 1 or sample.size == 0:
            raise ValueError("sample coordinate must be a non-empty one-dimensional ensemble axis")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("time coordinate must be a non-empty one-dimensional axis")
        if latitudes.size != self.latitudes.size or not np.allclose(latitudes, self.latitudes):
            raise ValueError("Dataset lat coordinate does not match adapter global latitudes")
        if longitudes.size != self.longitudes.size or not np.allclose(
            np.mod(longitudes, 360.0), np.mod(self.longitudes, 360.0)
        ):
            raise ValueError("Dataset lon coordinate does not match adapter global longitudes")
        if self.variables is not None and tuple(variables) != self.variables:
            raise ValueError("Dataset variables do not match adapter variables")
        if self.times is not None:
            expected_times = np.asarray(self.times)
            if expected_times.shape != times.shape or not np.array_equal(expected_times, times):
                raise ValueError("Dataset time coordinate does not match adapter times")
        expected_shape = (sample.size, times.size, latitudes.size, longitudes.size)
        coordinate_specs = {}
        for name, coord in getattr(dataset, "coords", {}).items():
            if name in {"sample", "time", "lat", "lon"}:
                continue
            dims = tuple(getattr(coord, "dims", ()))
            values = np.asarray(getattr(coord, "values", coord))
            if any(dim not in {"sample", "time"} for dim in dims):
                raise ValueError(
                    f"auxiliary coordinate {name!r} may only depend on sample/time, got {dims}"
                )
            expected_aux_shape = tuple(sample.size if dim == "sample" else times.size for dim in dims)
            if values.shape != expected_aux_shape:
                raise ValueError(f"auxiliary coordinate {name!r} shape does not match its dimensions")
            coordinate_specs[name] = (
                dims,
                np.array(values, copy=True),
                deepcopy(dict(getattr(coord, "attrs", {}) or {})),
            )
        arrays = {}
        for name, data_array in variables.items():
            values = _array_values(data_array, name)
            if values.shape != expected_shape:
                raise ValueError(f"variable {name!r} shape {values.shape} does not match {expected_shape}")
            arrays[str(name)] = values
        return arrays, sample, times, latitudes, longitudes, variables, coordinate_specs

    def regional_predict(
        self,
        global_predictions: Any,
        region: Optional[Mapping[str, Any] | RegionSpec] = None,
        target_latitudes: Optional[Sequence[float]] = None,
        target_longitudes: Optional[Sequence[float]] = None,
    ):
        arrays, sample, times, latitudes, longitudes, source_vars, coordinate_specs = self._validate_dataset(global_predictions)
        spec = _as_region(region)
        if (target_latitudes is None) != (target_longitudes is None):
            raise ValueError("target_latitudes and target_longitudes must be provided together")
        if spec is None and target_latitudes is not None:
            raise ValueError("region is required when requesting a target grid")
        if spec is None:
            lat_indices = np.arange(latitudes.size, dtype=np.int64)
            lon_indices = np.arange(longitudes.size, dtype=np.int64)
        else:
            lat_indices, lon_indices = _indices(
                latitudes,
                longitudes,
                spec,
                allow_empty=target_latitudes is not None,
            )
        out_lats = latitudes[lat_indices]
        out_lons = longitudes[lon_indices]
        target_lats = target_lons = None
        if target_latitudes is not None:
            target_lats = _target_axis(target_latitudes, "target_latitudes")
            target_lons = _target_axis(target_longitudes, "target_longitudes", wrap_longitude=True)
            lat_lo, lat_hi = sorted((spec.lat_min, spec.lat_max))
            if np.any(target_lats < lat_lo - 1e-7) or np.any(target_lats > lat_hi + 1e-7):
                raise ValueError("target_latitudes must lie inside the requested region")
            start = np.mod(spec.lon_min, 360.0)
            span = _region_span(spec)
            target_unwrapped = start + np.mod(np.mod(target_lons, 360.0) - start, 360.0)
            if span < 360.0 - 1e-9:
                requested_end = start + span
                if np.any(target_unwrapped < start - 1e-7) or np.any(target_unwrapped > requested_end + 1e-7):
                    raise ValueError("target_longitudes must lie inside the requested region")
            # Use all global longitudes for bracketing.  A target near the end
            # of a date-line region may lie between the selected endpoint and
            # the next global grid point just outside that region.
            all_indices = np.argsort(
                _longitude_interpolation_axis(longitudes, np.arange(longitudes.size), spec)
            )
            source_unwrapped = _longitude_interpolation_axis(longitudes, all_indices, spec)
            all_lat_indices = np.argsort(latitudes)
            source_latitudes = latitudes[all_lat_indices]
            out_lats = target_lats
            out_lons = target_lons
        result_vars = {}
        variable_attrs = {}
        for name, values in arrays.items():
            selected = np.take(np.take(values, lat_indices, axis=2), lon_indices, axis=3)
            if target_lats is not None:
                interpolation_values = np.take(np.take(values, all_lat_indices, axis=2), all_indices, axis=3)
                interpolation_values = _interp_axis(interpolation_values, source_latitudes, target_lats, axis=2)
                selected = _interp_axis(interpolation_values, source_unwrapped, target_unwrapped, axis=3)
            result_vars[name] = np.array(selected, copy=True)
            variable_attrs[name] = deepcopy(dict(getattr(source_vars[name], "attrs", {}) or {}))
        attrs = deepcopy(dict(getattr(global_predictions, "attrs", {}) or {}))
        coords = {
            "sample": (("sample",), np.array(sample, copy=True),
                       deepcopy(dict(getattr(global_predictions.coords["sample"], "attrs", {}) or {}))),
            "time": (("time",), np.array(times, copy=True),
                     deepcopy(dict(getattr(global_predictions.coords["time"], "attrs", {}) or {}))),
            "lat": (("lat",), np.array(out_lats, copy=True),
                    deepcopy(dict(getattr(global_predictions.coords["lat"], "attrs", {}) or {}))),
            "lon": (("lon",), np.array(out_lons, copy=True),
                    deepcopy(dict(getattr(global_predictions.coords["lon"], "attrs", {}) or {}))),
        }
        coords.update({
            name: (dims, np.array(values, copy=True), deepcopy(attrs))
            for name, (dims, values, attrs) in coordinate_specs.items()
        })
        return _make_dataset(result_vars, coords, attrs, variable_attrs)

    predict = regional_predict

    def infer(
        self,
        model_input: Any,
        region: Optional[Mapping[str, Any] | RegionSpec] = None,
        target_latitudes: Optional[Sequence[float]] = None,
        target_longitudes: Optional[Sequence[float]] = None,
        targets_template: Any = None,
        forcings: Any = None,
        rng: Any = None,
    ):
        if self.predictor is None:
            predictions = model_input
        else:
            try:
                inspect.signature(self.predictor).bind(
                    model_input,
                    targets_template=targets_template,
                    forcings=forcings,
                    rng=rng,
                )
            except (TypeError, ValueError):
                predictions = self.predictor(model_input)
            else:
                predictions = self.predictor(
                    model_input,
                    targets_template=targets_template,
                    forcings=forcings,
                    rng=rng,
                )
        return self.regional_predict(predictions, region, target_latitudes, target_longitudes)


__all__ = ["RegionSpec", "GenCastRegionalInference"]
