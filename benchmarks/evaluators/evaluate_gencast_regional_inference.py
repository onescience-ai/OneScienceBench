"""V2 evaluator adapter for GenCast regional inference."""
from __future__ import annotations

import ast, dataclasses, importlib.util, inspect, os, sys, traceback
from pathlib import Path
from typing import Dict
import numpy as np

_BASE_EVALUATOR_DIR = None
for _parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    _base_dir = _parent / 'benchmarks' / 'evaluators' / 'base'
    if _base_dir.is_dir():
        _BASE_EVALUATOR_DIR = _base_dir
        break
if _BASE_EVALUATOR_DIR is None:
    raise FileNotFoundError('Could not locate repository evaluators/base directory.')
sys.path.insert(0, str(_BASE_EVALUATOR_DIR))
from base_evaluator import *


def _repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / 'benchmarks' / 'datasets').is_dir():
            return candidate
    raise FileNotFoundError('Could not locate repository root containing benchmarks/datasets.')


candidate_env = os.environ.get("candidate_env")
stardard_name = os.environ.get("stardard_name")

def _candidate_model_path() -> Path:
    if not candidate_env:
        raise RuntimeError("candidate_env is not configured")
    candidate_path = os.environ.get(candidate_env)
    if not candidate_path:
        raise RuntimeError(f"{candidate_env} is not configured")
    return Path(candidate_path).expanduser().resolve()

def _reference_model_path() -> Path:
    return (
        _repo_root()
        / "benchmarks"
        / "standard"
        / stardard_name
    ).resolve()

CANDIDATE_DIR = _candidate_model_path()
REFERENCE_DIR = _reference_model_path()

REQUIRED_FILES = ('gencast_regional_inference.py',)
REQUIRED_CLASS_NAME = 'GenCastRegionalInference'
_DIMS = ('sample', 'time', 'lat', 'lon')
_VARIABLES = ('2m_temperature', '10m_u_component_of_wind')
_DATASET_ATTRS = {'model': 'GenCast', 'grid': 'global_regular', 'source': 'synthetic_global_forecast'}
_COORD_ATTRS = {'sample': {'long_name': 'ensemble member'}, 'time': {'standard_name': 'time'}, 'lat': {'standard_name': 'latitude', 'units': 'degrees_north'}, 'lon': {'standard_name': 'longitude', 'units': 'degrees_east'}}
_AUX_COORD_ATTRS = {'forecast_reference_time': {'standard_name': 'forecast_reference_time'}, 'lead_time': {'standard_name': 'forecast_period', 'units': 'hours'}, 'ensemble_label': {'long_name': 'ensemble member label'}, 'model_cycle': {'long_name': 'forecast production cycle'}}
_VAR_ATTRS = {'2m_temperature': {'units': 'K', 'standard_name': 'air_temperature'}, '10m_u_component_of_wind': {'units': 'm s-1', 'standard_name': 'eastward_wind'}}
_TIMES = np.array(['2024-01-01T00:00', '2024-01-01T06:00', '2024-01-01T12:00'], dtype='datetime64[ns]')


def _candidate_path(relative_path) -> Path:
    path = (CANDIDATE_DIR / relative_path).resolve()
    try:
        path.relative_to(CANDIDATE_DIR)
    except ValueError:
        raise ValueError(f'candidate path escapes CANDIDATE_DIR: {relative_path}')
    return path


def _source_path(config: Dict) -> Path:
    return _candidate_path(config.get('source_file', REQUIRED_FILES[0]))


def _load_candidate(config: Dict):
    path = _source_path(config)
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location('gencast_submission', str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['gencast_submission'] = module
    spec.loader.exec_module(module)
    return module


def _make_dataset(data_vars, coords, attrs):
    import xarray as xr
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def _values(value):
    return np.asarray(getattr(value, 'values', value))


def _dataset_signature(ds):
    if not hasattr(ds, 'data_vars') or not hasattr(ds, 'coords'):
        raise AssertionError('must return xarray-compatible Dataset')
    names = tuple(ds.data_vars.keys())
    if not names:
        raise AssertionError('Dataset must contain data variables')
    for name in ('sample', 'time', 'lat', 'lon'):
        if name not in ds.coords:
            raise AssertionError(f'missing coordinate {name}')
    coords = {name: (_values(coord), tuple(getattr(coord, 'dims', ())), dict(getattr(coord, 'attrs', {}) or {})) for name, coord in ds.coords.items()}
    data = {}
    for name in names:
        array = ds.data_vars[name]
        dims = tuple(getattr(array, 'dims', ()))
        if dims != _DIMS:
            raise AssertionError(f'{name} dims must be {_DIMS}, got {dims}')
        data[name] = (_values(array), dict(getattr(array, 'attrs', {}) or {}))
    return {'data': data, 'coords': coords, 'attrs': dict(getattr(ds, 'attrs', {}) or {})}


def _assert_dataset(actual, expected, label):
    actual_signature = _dataset_signature(actual)
    expected_signature = _dataset_signature(expected)
    assert actual_signature['attrs'] == expected_signature['attrs'], f'{label}: attrs not preserved'
    assert set(actual_signature['coords']) == set(expected_signature['coords']), f'{label}: coord names not preserved'
    for name in expected_signature['coords']:
        actual_values, actual_dims, actual_attrs = actual_signature['coords'][name]
        expected_values, expected_dims, expected_attrs = expected_signature['coords'][name]
        assert actual_dims == expected_dims, f'{label}: coord {name} dims mismatch'
        assert np.array_equal(actual_values, expected_values), f'{label}: coord {name} values mismatch'
        assert actual_attrs == expected_attrs, f'{label}: coord {name} attrs mismatch'
    assert tuple(actual_signature['data']) == tuple(expected_signature['data']), f'{label}: variable order not preserved'
    for name in expected_signature['data']:
        actual_values, actual_attrs = actual_signature['data'][name]
        expected_values, expected_attrs = expected_signature['data'][name]
        assert actual_values.shape == expected_values.shape, f'{label}: {name} shape mismatch'
        assert np.allclose(actual_values, expected_values, atol=1e-6, rtol=1e-6), f'{label}: {name} values differ'
        assert actual_attrs == expected_attrs, f'{label}: {name} attrs mismatch'


def _make_fixture(latitudes, longitudes, signed=False):
    sample = np.array([0, 1], dtype=np.int64)
    times = _TIMES.copy()
    arrays = {}
    for variable_index, name in enumerate(_VARIABLES):
        values = np.empty((sample.size, times.size, len(latitudes), len(longitudes)), dtype=np.float64)
        for member in range(sample.size):
            for time_index in range(times.size):
                for latitude_index, latitude in enumerate(latitudes):
                    for longitude_index, longitude in enumerate(longitudes):
                        physical_longitude = float(longitude) if signed else float(np.mod(longitude, 360.0))
                        values[member, time_index, latitude_index, longitude_index] = variable_index * 1_000_000 + member * 100_000 + time_index * 10_000 + float(latitude) * 100 + physical_longitude
        arrays[name] = (_DIMS, values, _VAR_ATTRS[name])
    coords = {
        'sample': (_DIMS[:1], sample, _COORD_ATTRS['sample']),
        'time': (_DIMS[1:2], times, _COORD_ATTRS['time']),
        'lat': (_DIMS[2:3], np.asarray(latitudes, dtype=float), _COORD_ATTRS['lat']),
        'lon': (_DIMS[3:4], np.asarray(longitudes, dtype=float), _COORD_ATTRS['lon']),
        'forecast_reference_time': ((), np.array(np.datetime64('2024-01-01T00:00', 'm')), _AUX_COORD_ATTRS['forecast_reference_time']),
        'lead_time': ((_DIMS[1],), np.array([0, 6, 12], dtype='timedelta64[h]'), _AUX_COORD_ATTRS['lead_time']),
        'ensemble_label': ((_DIMS[0],), np.array(['member_00', 'member_01']), _AUX_COORD_ATTRS['ensemble_label']),
        'model_cycle': ((), np.array('00Z'), _AUX_COORD_ATTRS['model_cycle']),
    }
    return _make_dataset(arrays, coords, _DATASET_ATTRS)


def _expected_region(ds, region, target_lats=None, target_lons=None):
    source = _dataset_signature(ds)
    latitudes = source['coords']['lat'][0]
    longitudes = source['coords']['lon'][0]
    latitude_low, latitude_high = sorted((region['lat_min'], region['lat_max']))
    latitude_indices = np.flatnonzero((latitudes >= latitude_low - 1e-9) & (latitudes <= latitude_high + 1e-9))
    wrapped_longitudes = np.mod(longitudes, 360.0)
    start, end = np.mod(region['lon_min'], 360.0), np.mod(region['lon_max'], 360.0)
    if abs(region['lon_max'] - region['lon_min']) >= 360.0 - 1e-9:
        longitude_indices = np.arange(longitudes.size, dtype=np.int64)
    elif start <= end:
        longitude_indices = np.flatnonzero((wrapped_longitudes >= start - 1e-9) & (wrapped_longitudes <= end + 1e-9))
    else:
        longitude_indices = np.concatenate((np.flatnonzero(wrapped_longitudes >= start - 1e-9), np.flatnonzero(wrapped_longitudes <= end + 1e-9)))
    output_latitudes, output_longitudes = latitudes[latitude_indices], longitudes[longitude_indices]
    if target_lats is not None:
        target_lats = np.asarray(target_lats, dtype=float)
        target_lons = np.asarray(target_lons, dtype=float)
        all_latitude_indices = np.argsort(latitudes)
        source_latitudes = latitudes[all_latitude_indices]
        all_longitude_indices = np.argsort(start + np.mod(wrapped_longitudes - start, 360.0))
        source_longitudes = start + np.mod(wrapped_longitudes[all_longitude_indices] - start, 360.0)
        target_longitudes_unwrapped = start + np.mod(np.mod(target_lons, 360.0) - start, 360.0)
        output_latitudes, output_longitudes = target_lats, target_lons
    arrays = {}
    for name in _VARIABLES:
        values = source['data'][name][0]
        if target_lats is not None:
            values = np.take(np.take(values, all_latitude_indices, axis=2), all_longitude_indices, axis=3)
            def interpolate(values, source_coordinates, target_coordinates, axis):
                source_coordinates = np.asarray(source_coordinates, dtype=float)
                if source_coordinates[0] > source_coordinates[-1]:
                    source_coordinates = source_coordinates[::-1]
                    values = np.flip(values, axis=axis)
                moved = np.moveaxis(values, axis, -1)
                flat = moved.reshape(-1, moved.shape[-1])
                result = np.stack([np.interp(target_coordinates, source_coordinates, row) for row in flat], axis=0)
                return np.moveaxis(result.reshape(moved.shape[:-1] + (target_coordinates.size,)), -1, axis)
            values = interpolate(values, source_latitudes, target_lats, axis=2)
            values = interpolate(values, source_longitudes, target_longitudes_unwrapped, axis=3)
        else:
            values = np.take(np.take(values, latitude_indices, axis=2), longitude_indices, axis=3)
        arrays[name] = (_DIMS, values, _VAR_ATTRS[name])
    coords = {
        'sample': (_DIMS[:1], source['coords']['sample'][0].copy(), _COORD_ATTRS['sample']),
        'time': (_DIMS[1:2], source['coords']['time'][0].copy(), _COORD_ATTRS['time']),
        'lat': (_DIMS[2:3], np.array(output_latitudes, copy=True), _COORD_ATTRS['lat']),
        'lon': (_DIMS[3:4], np.array(output_longitudes, copy=True), _COORD_ATTRS['lon']),
    }
    for name in source['coords']:
        if name not in coords:
            values, dims, attrs = source['coords'][name]
            coords[name] = (dims, np.array(values, copy=True), attrs)
    return _make_dataset(arrays, coords, _DATASET_ATTRS)


def _make_instance(cls, latitudes, longitudes, predictor=None):
    common = {'latitudes': latitudes, 'longitudes': longitudes, 'variables': _VARIABLES, 'times': _TIMES}
    if predictor is not None:
        try:
            return cls(**common, predictor=predictor)
        except TypeError:
            return cls(**common, model=predictor)
    return cls(**common)


def _offset_dataset(ds, offset):
    signature = _dataset_signature(ds)
    data_vars = {name: (_DIMS, values + offset, attrs) for name, (values, attrs) in signature['data'].items()}
    coords = {name: (dims, np.array(values, copy=True), attrs) for name, (values, dims, attrs) in signature['coords'].items()}
    return _make_dataset(data_vars, coords, signature['attrs'])


def _dataset_from_signature(signature, data_vars=None, coords=None):
    if data_vars is None:
        data_vars = {name: (_DIMS, np.array(values, copy=True), attrs) for name, (values, attrs) in signature['data'].items()}
    if coords is None:
        coords = {name: (dims, np.array(values, copy=True), attrs) for name, (values, dims, attrs) in signature['coords'].items()}
    return _make_dataset(data_vars, coords, signature['attrs'])


def _make_region(module, bounds):
    region_type = getattr(module, 'RegionSpec', None)
    if region_type is None:
        raise AssertionError('RegionSpec is required')
    try:
        return region_type(**bounds)
    except TypeError:
        return region_type(*(bounds[name] for name in ('lat_min', 'lat_max', 'lon_min', 'lon_max')))


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = ['file:gencast_regional_inference.py']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        required_files = tuple(config.get('required_files', REQUIRED_FILES))
        path = _candidate_path(required_files[0])
        update_check(current_index, path.is_file(), str(path))
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = ['checkpoint:not_required']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        detail = config.get('reason', 'not applicable: GenCast adapter uses synthetic predictor, no checkpoint')
        update_check(current_index, True, detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = ['eval:class_exists', 'eval:has_regional_predict', 'eval:has_infer', 'eval:has_init', 'eval:init_accepts_global_contract', 'eval:has_region_spec', 'eval:infer_callable', 'eval:regional_predict_callable']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        sub_path = _source_path(config)
        tree = ast.parse(sub_path.read_text(encoding='utf-8'), filename=str(sub_path))
        train_module = _load_candidate(config)
        class_name = config.get('class_name', REQUIRED_CLASS_NAME)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        target = next((node for node in classes if node.name == class_name), None)
        update_check(current_index, target is not None, class_name)
        current_index += 1
        if target:
            methods = {node.name for node in target.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            update_check(current_index, 'regional_predict' in methods, str(sorted(methods)))
            current_index += 1
            update_check(current_index, 'infer' in methods, str(sorted(methods)))
            current_index += 1
            init = next((node for node in target.body if isinstance(node, ast.FunctionDef) and node.name == '__init__'), None)
            update_check(current_index, init is not None, str(bool(init)))
            current_index += 1
            if init:
                names = {argument.arg for argument in init.args.args}
                required = {'latitudes', 'longitudes', 'variables', 'times'}
                update_check(current_index, required.issubset(names), str(sorted(names)))
            current_index += 1
        else:
            current_index += 4
        region_type = getattr(train_module, config.get('required_module_class', 'RegionSpec'), None)
        update_check(current_index, region_type is not None, str(type(region_type)))
        current_index += 1
        model_cls = getattr(train_module, class_name, None)
        infer = getattr(model_cls, 'infer', None)
        regional = getattr(model_cls, 'regional_predict', None)
        update_check(current_index, callable(infer), str(inspect.signature(infer)) if callable(infer) else 'missing')
        current_index += 1
        update_check(current_index, callable(regional), str(inspect.signature(regional)) if callable(regional) else 'missing')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = ['model:init_global_axes', 'model:region_spec_exists', 'model:region_spec_immutable', 'model:stores_global_axes']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        train_module = _load_candidate(config)
        model_cls = getattr(train_module, config.get('class_name', REQUIRED_CLASS_NAME), None)
        required = ('latitudes', 'longitudes', 'variables', 'times')
        init = inspect.signature(model_cls.__init__) if inspect.isclass(model_cls) else None
        missing = [name for name in required if init is None or name not in init.parameters]
        update_check(current_index, not missing, f'missing={missing}')
        current_index += 1
        region_type = getattr(train_module, config.get('region_spec_class', 'RegionSpec'), None)
        update_check(current_index, inspect.isclass(region_type), str(region_type))
        current_index += 1
        if region_type is not None:
            sample = region_type(0.0, 1.0, 0.0, 1.0)
            try:
                sample.lat_min = 2.0
            except (AttributeError, TypeError, dataclasses.FrozenInstanceError):
                frozen = True
            else:
                frozen = False
            update_check(current_index, frozen, 'frozen dataclass')
        current_index += 1
        model = _make_instance(model_cls, np.array([60., 0., -60.]), np.array([0., 90., 180., 270.]))
        update_check(current_index, all(hasattr(model, name) for name in ('latitudes', 'longitudes', 'variables', 'times')), 'adapter state')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = ['forward:returns_dataset', 'forward:variable_order', 'forward:named_dims', 'forward:finite', 'forward:region_shape']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        latitudes = np.array([60., 0., -60.])
        longitudes = np.array([0., 90., 180., 270.])
        dataset = _make_fixture(latitudes, longitudes)
        train_module = _load_candidate(config)
        model_cls = getattr(train_module, config.get('class_name', REQUIRED_CLASS_NAME))
        model = _make_instance(model_cls, latitudes, longitudes)
        region = config.get('call', {}).get('region', {'lat_min': 0., 'lat_max': 60., 'lon_min': 0., 'lon_max': 180.})
        output = model.regional_predict(dataset, region)
        signature = _dataset_signature(output)
        update_check(current_index, hasattr(output, 'data_vars') and hasattr(output, 'coords'), str(type(output)))
        current_index += 1
        update_check(current_index, tuple(signature['data']) == _VARIABLES, str(tuple(signature['data'])))
        current_index += 1
        update_check(current_index, all(tuple(array.dims) == _DIMS for array in output.data_vars.values()), str(signature['data']))
        current_index += 1
        update_check(current_index, all(np.isfinite(values).all() for values, _ in signature['data'].values()), 'all variables finite')
        current_index += 1
        update_check(current_index, signature['data'][_VARIABLES[0]][0].shape[2:] == (2, 3), str(signature['data'][_VARIABLES[0]][0].shape))
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = ['oracle:constructor_rejects_non-monotonic_lat', 'oracle:constructor_rejects_duplicate_lon', 'oracle:constructor_rejects_non-increasing_time', 'oracle:region_spec_immutable', 'oracle:regional_predict_normal', 'oracle:regional_predict_date_line', 'oracle:regional_predict_descending_lat', 'oracle:full_global', 'oracle:equivalent_longitude', 'oracle:target_grid', 'oracle:input_immutability', 'oracle:rejects_missing_coordinate', 'oracle:rejects_wrong_dimensions', 'oracle:rejects_forecast_time_mismatch', 'oracle:rejects_spatial_shape_mismatch', 'oracle:rejects_non_numeric_variable', 'oracle:rejects_variable_order', 'oracle:predictor_global_then_region', 'oracle:predictor_single_argument', 'oracle:rejects_predictor_non_dataset', 'oracle:rejects_predictor_wrong_variable', 'oracle:edge_invalid_latitude_region', 'oracle:edge_one_sided_target_grid', 'oracle:edge_empty_grid_region']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        sub = _load_candidate(config)
        class_name = config.get('class_name', REQUIRED_CLASS_NAME)
        model_class = getattr(sub, class_name)
        latitudes = np.array([60.0, 30.0, 0.0, -30.0])
        longitudes = np.array([0.0, 90.0, 180.0, 270.0])
        global_input = _make_fixture(latitudes, longitudes)
        original = global_input.copy()
        sub_model = _make_instance(model_class, latitudes, longitudes)
        constructor_args = {'latitudes': latitudes, 'longitudes': longitudes, 'variables': _VARIABLES, 'times': _TIMES}
        for label, override in [('non-monotonic lat', {'latitudes': np.array([60.0, 0.0, 30.0, -30.0])}), ('duplicate lon', {'longitudes': np.array([0.0, 90.0, 180.0, 360.0])}), ('non-increasing time', {'times': _TIMES[::-1]})]:
            try:
                model_class(**{**constructor_args, **override})
            except (ValueError, KeyError, TypeError):
                ok = True
            else:
                ok = False
            update_check(current_index, ok, label)
            current_index += 1
        try:
            region = sub.RegionSpec(0.0, 10.0, 0.0, 10.0)
            region.lat_min = 1.0
        except (AttributeError, TypeError, dataclasses.FrozenInstanceError):
            ok = True
        else:
            ok = False
        update_check(current_index, ok, 'RegionSpec frozen')
        current_index += 1
        for label, region in [('normal', {'lat_min': 0, 'lat_max': 60, 'lon_min': 0, 'lon_max': 180}), ('date_line', {'lat_min': -30, 'lat_max': 30, 'lon_min': 270, 'lon_max': 30}), ('descending_lat', {'lat_min': 0, 'lat_max': 30, 'lon_min': 90, 'lon_max': 270})]:
            try:
                expected = _expected_region(global_input, region)
                _assert_dataset(sub_model.regional_predict(global_input, region=_make_region(sub, region)), expected, label)
                ok, detail = True, 'ok'
            except AssertionError as error:
                ok, detail = False, str(error)[:120]
            update_check(current_index, ok, detail)
            current_index += 1
        try:
            _assert_dataset(sub_model.regional_predict(global_input), global_input, 'global')
            ok, detail = True, 'ok'
        except AssertionError as error:
            ok, detail = False, str(error)[:120]
        update_check(current_index, ok, detail)
        current_index += 1
        equivalent_longitudes = np.array([0.0, 90.0, -180.0, -90.0])
        equivalent_input = _make_fixture(latitudes, equivalent_longitudes, signed=True)
        try:
            _assert_dataset(sub_model.regional_predict(equivalent_input), equivalent_input, 'eq_lon')
            ok, detail = True, 'ok'
        except AssertionError as error:
            ok, detail = False, str(error)[:120]
        update_check(current_index, ok, detail)
        current_index += 1
        target_region = {'lat_min': 0, 'lat_max': 60, 'lon_min': 0, 'lon_max': 180}
        target_spec = _make_region(sub, target_region)
        target_latitudes = np.array([15.0, 45.0])
        target_longitudes = np.array([45.0, 135.0])
        try:
            expected = _expected_region(global_input, target_region, target_latitudes, target_longitudes)
            _assert_dataset(sub_model.regional_predict(global_input, target_spec, target_latitudes, target_longitudes), expected, 'target')
            ok, detail = True, 'ok'
        except AssertionError as error:
            ok, detail = False, str(error)[:120]
        update_check(current_index, ok, detail)
        current_index += 1
        try:
            _assert_dataset(global_input, original, 'immutability')
            ok, detail = True, 'ok'
        except AssertionError as error:
            ok, detail = False, str(error)[:120]
        update_check(current_index, ok, detail)
        current_index += 1
        signature = _dataset_signature(global_input)
        base_data = {name: (_DIMS, np.array(values, copy=True), attrs) for name, (values, attrs) in signature['data'].items()}
        base_coords = {name: (dims, np.array(values, copy=True), attrs) for name, (values, dims, attrs) in signature['coords'].items()}
        missing_coord = dict(base_coords)
        missing_coord.pop('lat')
        first_name = _VARIABLES[0]
        wrong_dims = dict(base_data)
        wrong_dims[first_name] = (('sample', 'time', 'lon', 'lat'), np.array(signature['data'][first_name][0], copy=True), signature['data'][first_name][1])
        wrong_time = dict(base_coords)
        wrong_time['time'] = (('time',), _TIMES + np.timedelta64(1, 'h'), _COORD_ATTRS['time'])
        short_data = {name: (_DIMS, values[:, :, :-1, :], attrs) for name, (values, attrs) in signature['data'].items()}
        short_coords = dict(base_coords)
        short_coords['lat'] = (('lat',), signature['coords']['lat'][0][:-1], signature['coords']['lat'][2])
        non_numeric = dict(base_data)
        non_numeric[first_name] = (_DIMS, signature['data'][first_name][0].astype(str), signature['data'][first_name][1])
        reversed_data = {name: base_data[name] for name in reversed(_VARIABLES)}
        invalid_datasets = [('missing_coordinate', _dataset_from_signature(signature, base_data, missing_coord)), ('wrong_dimensions', _dataset_from_signature(signature, wrong_dims, base_coords)), ('forecast_time_mismatch', _dataset_from_signature(signature, base_data, wrong_time)), ('spatial_shape_mismatch', _dataset_from_signature(signature, short_data, short_coords)), ('non_numeric_variable', _dataset_from_signature(signature, non_numeric, base_coords)), ('variable_order', _dataset_from_signature(signature, reversed_data, base_coords))]
        for label, invalid_dataset in invalid_datasets:
            try:
                sub_model.regional_predict(invalid_dataset)
            except (ValueError, KeyError, TypeError):
                ok = True
            else:
                ok = False
            update_check(current_index, ok, label)
            current_index += 1
        observed = {}
        targets_template, forcings, rng = object(), object(), object()
        def official_predictor(inputs, targets_template=None, forcings=None, rng=None):
            observed.update(inputs=inputs, targets_template=targets_template, forcings=forcings, rng=rng)
            return _offset_dataset(inputs, 7.5)
        predictor_region = {'lat_min': 0, 'lat_max': 60, 'lon_min': 0, 'lon_max': 180}
        official_model = _make_instance(model_class, latitudes, longitudes, official_predictor)
        try:
            actual = official_model.infer(global_input, predictor_region, targets_template=targets_template, forcings=forcings, rng=rng)
            expected = _expected_region(_offset_dataset(global_input, 7.5), predictor_region)
            _assert_dataset(actual, expected, 'predictor_order')
            ok = observed.get('inputs') is global_input and observed.get('targets_template') is targets_template and observed.get('forcings') is forcings and observed.get('rng') is rng
            detail = 'full global Dataset and official predictor arguments preserved'
        except (AssertionError, ValueError, TypeError) as error:
            ok, detail = False, str(error)[:200]
        update_check(current_index, ok, detail)
        current_index += 1
        one_argument_seen = []
        def one_argument_predictor(inputs):
            one_argument_seen.append(inputs)
            return inputs
        try:
            one_argument_model = _make_instance(model_class, latitudes, longitudes, one_argument_predictor)
            _assert_dataset(one_argument_model.infer(global_input), global_input, 'one_arg_predictor')
            ok, detail = bool(one_argument_seen) and one_argument_seen[0] is global_input, 'single-input predictor fallback'
        except (AssertionError, ValueError, TypeError) as error:
            ok, detail = False, str(error)[:200]
        update_check(current_index, ok, detail)
        current_index += 1
        invalid_predictors = [('non_dataset', lambda inputs: object()), ('wrong_variable', lambda inputs: _make_dataset({'unexpected': (_DIMS, np.zeros((2, 3, 4, 4)), {})}, {'sample': ('sample', [0, 1], {}), 'time': ('time', _TIMES, {}), 'lat': ('lat', [60, 30, 0, -30], {}), 'lon': ('lon', [0, 90, 180, 270], {})}, {}))]
        for label, predictor in invalid_predictors:
            try:
                _make_instance(model_class, latitudes, longitudes, predictor).infer(global_input)
            except (ValueError, KeyError, TypeError, AssertionError):
                ok = True
            else:
                ok = False
            update_check(current_index, ok, label)
            current_index += 1
        edge_cases = [('invalid_latitude_region', lambda: sub_model.regional_predict(global_input, _make_region(sub, {'lat_min': 95, 'lat_max': 100, 'lon_min': 0, 'lon_max': 10}))), ('one_sided_target_grid', lambda: sub_model.regional_predict(global_input, target_spec, target_latitudes=[15.0], target_longitudes=None)), ('empty_grid_region', lambda: sub_model.regional_predict(global_input, _make_region(sub, {'lat_min': 10, 'lat_max': 20, 'lon_min': 10, 'lon_max': 20})))]
        for label, fn in edge_cases:
            try:
                fn()
            except (ValueError, KeyError, TypeError):
                ok = True
            else:
                ok = False
            update_check(current_index, ok, label)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
