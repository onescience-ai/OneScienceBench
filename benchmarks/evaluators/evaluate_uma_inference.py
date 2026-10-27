#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import importlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

# Base is deployed at benchmarks/evaluators/base/ in the SCNet repository.
# Import the existing public evaluator functions, without adding wrappers.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'base'))
from base_evaluator import (
    CANDIDATE_DIR, REFERENCE_DIR,
    _load_candidate_modules,
    _load_reference_modules,
    check_files,
    check_config as _base_check_config,
    check_interface as _base_check_interface,
    check_infer_structure as _base_check_infer_structure,
)
from model_structure_checker import check_model_structure_v2, checkpoints_to_tuples

CHECKPOINT_DIR = Path('/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint')
PREFERRED = ('uma-s-1p1_converted.pt', 'uma-s-1p1.pt')
REQUIRED_CONFIG = ('checkpoint', 'device', 'dtype', 'dataset', 'batch_size',
                   'lmax', 'mmax', 'norm_type', 'act_type', 'ff_type')
EXPECTED_CONFIG = {'lmax': 2, 'mmax': 2, 'norm_type': 'rms_norm_sh',
                   'act_type': 'gate', 'ff_type': 'spectral'}
SEED = 20260817
CACHE = {}


def check_config_content(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Reuse shared YAML checking and validate UMA's flat configuration contract."""
    results = []
    try:
        results.extend(_base_check_config(config, caller_globals=caller_globals,
                                          caller_locals=caller_locals))
    except Exception:
        results.append(('config:base_yaml', False, traceback.format_exc()))
    try:
        import yaml
        path = CANDIDATE_DIR / config.get('config_name', 'config.yaml')
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise TypeError('flat config.yaml must be a mapping')
        missing = sorted(set(REQUIRED_CONFIG) - set(data))
        nested = sorted(k for k, v in data.items() if isinstance(v, dict))
        forbidden = sorted(set(data).intersection({'model', 'backbone', 'heads'}))
        if missing or nested or forbidden:
            raise AssertionError(f'missing={missing}, nested={nested}, forbidden={forbidden}')
        results.append(('config:flat_required_keys', True, f'{len(data)} flat configuration keys'))
    except Exception:
        return results + [('config:flat_required_keys', False, traceback.format_exc())]
    for key, expected in EXPECTED_CONFIG.items():
        actual = data.get(key)
        results.append((f'config:{key}', actual == expected,
                        f'expected {expected!r}, got {actual!r}'))
    try:
        path = Path(str(data['checkpoint'])).expanduser()
        if path.resolve() != CHECKPOINT_DIR.resolve():
            raise AssertionError(f'default checkpoint must be {CHECKPOINT_DIR}, got {path}')
        if type(data['batch_size']) is not int or data['batch_size'] <= 0:
            raise ValueError('batch_size must be a positive integer')
        results.append(('config:checkpoint_batch', True, 'default directory and batch size valid'))
    except Exception:
        results.append(('config:checkpoint_batch', False, traceback.format_exc()))
    return results


def check_interface(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Delegate ordinary interfaces to base; verify the required UMA class/forward."""
    if config.get('phase') != 'model_structure':
        return list(_base_check_interface(config, caller_globals=caller_globals,
                                          caller_locals=caller_locals))
    try:
        tree = ast.parse((CANDIDATE_DIR / 'uma.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UMA')
        methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if 'forward' not in methods:
            raise AssertionError('UMA must implement forward(data)')
        checks = [('model_interface:class_forward', True, 'UMA.forward defined in uma.py')]
        mod = _load_candidate_modules('uma.py')
        model_type = getattr(mod, 'UMA', None)
        if not inspect.isclass(model_type) or not issubclass(model_type, torch.nn.Module):
            raise TypeError('UMA must be a subclass of torch.nn.Module')
        checks.append(('model_interface:torch_module', True, 'UMA(torch.nn.Module)'))
        return checks
    except Exception:
        return [('model_interface:class_forward', False, traceback.format_exc()),
                ('model_interface:torch_module', False, 'class validation failed')]


def check_infer_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Reuse base inference AST while rejecting import-time checkpoint/GPU/LMDB access."""
    checks = []
    try:
        common = _base_check_infer_structure(config, caller_globals=caller_globals,
                                             caller_locals=caller_locals)
        # Autograd is essential for gradient-derived force/stress; lack of
        # no_grad() is not an inference defect for UMA.
        checks.extend((n, p, d) for n, p, d in common if n not in {'inference_start:has_no_grad', 'inference_start:has_eval'})
    except Exception:
        checks.append(('infer_structure:base_ast', False, traceback.format_exc()))
    try:
        model_tree = ast.parse((CANDIDATE_DIR / 'uma.py').read_text(encoding='utf-8'))
        infer_tree = ast.parse((CANDIDATE_DIR / 'uma_infer.py').read_text(encoding='utf-8'))
        top_calls = []
        for node in infer_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If) and any(isinstance(k, ast.Constant) and k.value == '__main__'
                                                for k in ast.walk(node.test)):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                parts = []
                while isinstance(func, ast.Attribute):
                    parts.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    parts.append(func.id)
                name = '.'.join(reversed(parts))
                if name in {'torch.load', 'load_model', 'lmdb.open', 'torch.cuda.is_available',
                            'torch.cuda.device_count', 'torch.cuda.current_device'}:
                    top_calls.append(name)
        if top_calls:
            raise AssertionError(f'import-time side effects: {top_calls}')
        forward = next((n for c in model_tree.body if isinstance(c, ast.ClassDef) and c.name == 'UMA'
                        for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'), None)
        if forward is None:
            raise AssertionError('UMA.forward is missing')
        for n in ast.walk(forward):
            if isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute):
                        if item.context_expr.func.attr in {'no_grad', 'inference_mode'}:
                            raise AssertionError('UMA.forward must retain gradients')
        checks.append(('infer_structure:source_side_effects', True, 'no prohibited import-level calls or forward no_grad'))
    except Exception:
        checks.append(('infer_structure:source_side_effects', False, traceback.format_exc()))
    try:
        source = (CANDIDATE_DIR / 'uma_infer.py').read_text(encoding='utf-8').lower()
        required = {'ase_atoms': 'ase', 'lmdb': 'lmdb', 'batch': 'batch',
                    'numpy_detach': 'detach', 'numpy_conversion': 'numpy'}
        missing = [key for key, marker in required.items() if marker not in source]
        if missing:
            raise AssertionError(f'inference source lacks supported-input/conversion markers: {missing}')
        checks.append(('infer_structure:input_adapters', True,
                       'ASE, LMDB, batching and detach/cpu/numpy paths declared'))
    except Exception:
        checks.append(('infer_structure:input_adapters', False, traceback.format_exc()))
    try:
        old_path = sys.path[:]
        prev_uma = sys.modules.pop('uma', None)
        try:
            sys.path.insert(0, str(CANDIDATE_DIR))
            with mock.patch.object(torch, 'load', side_effect=RuntimeError('checkpoint access at import time')), \
                 mock.patch.object(torch.cuda, 'device_count', side_effect=RuntimeError('GPU query at import time')), \
                 mock.patch.object(torch.cuda, 'is_available', side_effect=RuntimeError('GPU query at import time')):
                try:
                    lmdb_mod = importlib.import_module('lmdb')
                except ModuleNotFoundError:
                    lmdb_mod = None
                if lmdb_mod is not None:
                    with mock.patch.object(lmdb_mod, 'open', side_effect=RuntimeError('LMDB access at import time')):
                        _load_candidate_modules('uma.py')
                        _load_candidate_modules('uma_infer.py')
                else:
                    _load_candidate_modules('uma.py')
                    _load_candidate_modules('uma_infer.py')
            checks.append(('infer_structure:lazy_import', True, 'import has no checkpoint/GPU/LMDB access'))
        finally:
            sys.path[:] = old_path
            sys.modules.pop('uma', None)
            if prev_uma is not None:
                sys.modules['uma'] = prev_uma
    except Exception:
        checks.append(('infer_structure:lazy_import', False, traceback.format_exc()))
    return checks


def check_checkpoint_file(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Enforce deterministic preferred checkpoint selection and unique .pt fallback."""
    try:
        directory = Path(config.get('checkpoint_dir', CHECKPOINT_DIR))
        if directory.resolve() != CHECKPOINT_DIR.resolve() or not directory.is_dir():
            raise FileNotFoundError(f'fixed checkpoint directory unavailable: {CHECKPOINT_DIR}')
        preferred = next((directory / n for n in PREFERRED if (directory / n).is_file()), None)
        all_pt = sorted(p for p in directory.glob('*.pt') if p.is_file())
        if preferred is None:
            if len(all_pt) != 1:
                raise FileNotFoundError(f'fallback requires exactly one *.pt, found {len(all_pt)}')
            preferred = all_pt[0]
        if not preferred.stat().st_size:
            raise AssertionError('checkpoint file is empty')
        CACHE.clear()
        CACHE['checkpoint_path'] = preferred
        return [('checkpoint_file:priority_resolution', True, f'{preferred.name} ({preferred.stat().st_size} bytes)')]
    except Exception:
        return [('checkpoint_file:priority_resolution', False, traceback.format_exc())]


def check_checkpoint_loading(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Validate both checkpoint schema alternatives, CPU loading and strict state equality."""
    names = ('checkpoint:payload_schema', 'checkpoint:checkpoint_config',
             'checkpoint:strict_tensor_equality', 'checkpoint:cpu_load_and_dtype',
             'checkpoint:no_unexpected_weights')
    try:
        checkpoint_path = CACHE.get('checkpoint_path')
        if checkpoint_path is None:
            ck = check_checkpoint_file({'checkpoint_dir': str(CHECKPOINT_DIR)})
            if not ck[0][1]:
                raise RuntimeError(ck[0][2])
            checkpoint_path = CACHE['checkpoint_path']
        # Prefer tensor-only deserialization. The fixed checkpoint contains
        # a legacy class path under onescience.utils.uma.units.mlip_unit.api.
        # Create that import path locally rather than requiring OneScience to
        # be installed in the evaluator environment.
        load_mode = 'weights_only=True'
        try:
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception as safe_error:
            legacy_module_name = 'onescience.utils.uma.units.mlip_unit.api.inference'
            package_names = (
                'onescience',
                'onescience.utils',
                'onescience.utils.uma',
                'onescience.utils.uma.units',
                'onescience.utils.uma.units.mlip_unit',
                'onescience.utils.uma.units.mlip_unit.api',
            )
            for package_name in package_names:
                package = sys.modules.get(package_name)
                if package is None:
                    package = types.ModuleType(package_name)
                    sys.modules[package_name] = package
                if not hasattr(package, '__path__'):
                    package.__path__ = []
                parent_name, _, child_name = package_name.rpartition('.')
                if parent_name and parent_name in sys.modules:
                    setattr(sys.modules[parent_name], child_name, package)

            legacy_module = sys.modules.get(legacy_module_name)
            if legacy_module is None:
                legacy_module = types.ModuleType(legacy_module_name)
                sys.modules[legacy_module_name] = legacy_module
            setattr(sys.modules['onescience.utils.uma.units.mlip_unit.api'],
                    'inference', legacy_module)
            if not hasattr(legacy_module, '__getattr__'):
                legacy_module.__getattr__ = lambda name: (
                    legacy_module.__dict__.setdefault(
                        name,
                        type(name, (), {'__module__': legacy_module_name})
                    ) if not name.startswith('__') else
                    (_ for _ in ()).throw(AttributeError(name))
                )
            legacy_cls = getattr(legacy_module, 'MLIPInferenceCheckpoint')
            # Newer PyTorch versions support explicitly allowlisting the
            # trusted checkpoint container for a second safe-load attempt.
            add_safe_globals = getattr(torch.serialization, 'add_safe_globals', None)
            if callable(add_safe_globals):
                try:
                    add_safe_globals([legacy_cls])
                except Exception:
                    pass
            try:
                payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                load_mode = 'weights_only=True with legacy checkpoint class allowlisted'
            except Exception:
                try:
                    # This fallback is limited to the fixed checkpoint selected
                    # from CHECKPOINT_DIR, a trusted benchmark asset.
                    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    load_mode = 'trusted fixed-checkpoint legacy pickle compatibility'
                except Exception as legacy_error:
                    raise RuntimeError(
                        f'weights_only=True failed ({type(safe_error).__name__}: {safe_error}); '
                        f'legacy compatibility load failed '
                        f'({type(legacy_error).__name__}: {legacy_error})'
                    ) from legacy_error

        # Some converted checkpoints serialize a lightweight container object
        # instead of a raw dict. Its public fields carry the same schema.
        if not isinstance(payload, Mapping):
            if hasattr(payload, '__dict__'):
                payload = vars(payload)
            else:
                raise TypeError(
                    f'checkpoint must be a mapping or attribute container, got {type(payload).__name__}'
                )
        raw_model_config = payload.get('model_config')
        raw_model_state = payload.get('model_state_dict')
        if not isinstance(raw_model_config, Mapping) and hasattr(raw_model_config, '__dict__'):
            raw_model_config = vars(raw_model_config)
        if isinstance(raw_model_config, Mapping) and isinstance(raw_model_state, Mapping):
            model_config, state, format_name = raw_model_config, raw_model_state, 'model_config/model_state_dict'
        elif payload.get('config') is not None and isinstance(payload.get('state_dict'), Mapping):
            saved = payload['config']
            if not isinstance(saved, Mapping) and hasattr(saved, '__dict__'):
                saved = vars(saved)
            if not isinstance(saved, Mapping):
                raise TypeError('checkpoint config must be a mapping or attribute container')
            model_config = saved.get('model', saved)
            if not isinstance(model_config, Mapping) and hasattr(model_config, '__dict__'):
                model_config = vars(model_config)
            state, format_name = payload['state_dict'], 'config/state_dict'
        else:
            raise KeyError('expected (model_config, model_state_dict) or (config, state_dict)')
        if not isinstance(model_config, Mapping) or not isinstance(state, Mapping) or not state:
            raise TypeError('invalid checkpoint model configuration or empty state')
        if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise TypeError('checkpoint state must be a string-to-tensor mapping')
        CACHE.update(payload=payload, state=state, model_config=model_config)
        checks = [(names[0], True, f'{format_name}, {len(state)} parameter/buffer tensors; {load_mode}')]
    except Exception:
        return [(n, False, traceback.format_exc()) for n in names]
    try:
        mod = _load_candidate_modules('uma.py')
        cls = getattr(mod, 'UMA')
        if not inspect.isclass(cls) or not issubclass(cls, torch.nn.Module):
            raise TypeError('uma.py must define a real torch.nn.Module UMA')
        sig = inspect.signature(cls)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kwargs = dict(model_config) if accepts_kwargs else {k: v for k, v in model_config.items() if k in sig.parameters}
        # Some implementations accept a whole checkpoint model_config object.
        if not kwargs and 'model_config' in sig.parameters:
            kwargs = {'model_config': dict(model_config)}
        constructed = cls(**kwargs)
        if set(constructed.state_dict()) != set(state):
            raise AssertionError('UMA instantiated from checkpoint model_config has different state keys')
        checks.append((names[1], True, 'checkpoint model_config builds matching UMA structure'))
    except Exception:
        checks.append((names[1], False, traceback.format_exc()))
    try:
        infer = _load_candidate_modules('uma_infer.py')
        func = getattr(infer, 'load_model')
        sig = inspect.signature(func)
        accept_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kwargs = {}
        if 'checkpoint_path' in sig.parameters or accept_var:
            kwargs['checkpoint_path'] = str(checkpoint_path)
        elif 'checkpoint' in sig.parameters:
            kwargs['checkpoint'] = str(checkpoint_path)
        else:
            raise TypeError('load_model must accept a checkpoint_path/checkpoint argument')
        if 'device' in sig.parameters or accept_var:
            kwargs['device'] = 'cpu'
        if 'dtype' in sig.parameters or accept_var:
            kwargs['dtype'] = config.get('dtype', 'float32')
        with mock.patch.object(torch, 'load', wraps=torch.load) as load_spy, \
             mock.patch.object(torch.nn.Module, 'load_state_dict', autospec=True,
                               side_effect=torch.nn.Module.load_state_dict) as strict_spy:
            model = func(**kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError('load_model must return an nn.Module')
        if not any(
            str(c.kwargs.get('map_location', c.args[1] if len(c.args) > 1 else None))
            in ('cpu', "device(type='cpu')")
            for c in load_spy.call_args_list
        ):
            raise AssertionError('checkpoint must first be loaded with CPU map_location')
        strict_full_state = False
        for call in strict_spy.call_args_list:
            loaded_state = call.args[1] if len(call.args) > 1 else call.kwargs.get('state_dict')
            strict_value = call.kwargs.get('strict', call.args[2] if len(call.args) > 2 else True)
            if strict_value is True and isinstance(loaded_state, Mapping) and set(loaded_state) == set(state):
                strict_full_state = True
                break
        if not strict_full_state:
            raise AssertionError(
                'candidate must call load_state_dict(strict=True) with the complete checkpoint state '
                '(positional or keyword state_dict is accepted)'
            )
        actual = model.state_dict()
        if set(actual) != set(state) or len(actual) != len(state):
            raise AssertionError('model state differs from full checkpoint key set')
        for k, v in state.items():
            if actual[k].shape != v.shape:
                raise AssertionError(f'{k} shape mismatch {actual[k].shape} != {v.shape}')
            if not torch.equal(actual[k].detach().cpu(), v.detach().cpu().to(actual[k].dtype)):
                raise AssertionError(f'{k} checkpoint value mismatch')
        model.cpu().eval()
        CACHE.update(model=model, infer=infer, model_module=mod)
        checks.append((names[2], True, f'{len(state)} full state tensors loaded exactly, strict=True observed'))
        checks.append((names[3], True, 'checkpoint loaded CPU-first, model returned on CPU'))
        checks.append((names[4], True, 'all checkpoint parameter names and values preserved'))
    except Exception:
        checks.extend((n, False, traceback.format_exc()) for n in names[2:])
    return checks


def check_model_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Shared declaration rules plus model architecture / semantic module checks."""
    if 'model' not in CACHE:
        loaded = check_checkpoint_loading({'dtype': 'float32'})
        if any(not passed for _, passed, _ in loaded):
            return [('model_structure:loaded_model', False, 'checkpoint load failed: ' + repr(loaded))]
    model = CACHE['model']
    checks = []
    try:
        rules = config.get('rules', [])
        if not rules:
            raise ValueError('model_structure.rules is empty')
        checks.extend(checkpoints_to_tuples(check_model_structure_v2(
            model, config_dict={'rules': rules}, verbose=False)))
    except Exception:
        checks.append(('model_structure:shared_rules', False, traceback.format_exc()))
    try:
        modules = list(model.named_modules())
        lower = ' '.join((name + ' ' + type(m).__name__).lower() for name, m in modules)
        # A semantic signature is intentionally used instead of requiring the
        # exact registered path hierarchy of HydraModel or a fixed module name.
        semantic = {
            'atomic_embedding': ('embedd', 'atomic'),
            'charge_spin': ('charge', 'spin'),
            'dataset_embedding': ('dataset', 'embedd'),
            'composition': ('compos',),
            'gaussian_rbf': ('gauss',),
            'cutoff': ('cutoff',),
            'so2_so3': ('so2', 'so3'),
            'equivariant_message': ('block',),
            'rms_normalization': ('rms',),
            'gated_activation': ('gate',),
            'spectral_feed_forward': ('spectral',),
            'mixture_of_experts': ('expert',),
            'energy_heads': ('head',),
        }
        source = (CANDIDATE_DIR / 'uma.py').read_text(encoding='utf-8').lower()
        for feature, terms in semantic.items():
            present = all(t in lower or t in source for t in terms)
            checks.append((f'model_structure:{feature}', present,
                           f'feature terms={terms}, registered modules={len(modules)}'))
    except Exception:
        checks.append(('model_structure:semantic_components', False, traceback.format_exc()))
    try:
        state = CACHE['state']
        actual = model.state_dict()
        if {k: tuple(v.shape) for k, v in actual.items()} != {k: tuple(v.shape) for k, v in state.items()}:
            raise AssertionError('full checkpoint state shape/key signature differs')
        parameters = list(model.parameters())
        if not parameters or sum(p.numel() for p in parameters) == 0:
            raise AssertionError('UMA model has no learnable parameters')
        checks.append(('model_structure:state_signature', True,
                       f'{len(actual)} keys, {sum(p.numel() for p in parameters)} parameters'))
    except Exception:
        checks.append(('model_structure:state_signature', False, traceback.format_exc()))
    for key, expected in EXPECTED_CONFIG.items():
        model_value = CACHE.get('model_config', {}).get(key, None)
        # lmax etc may be nested one level under backbone in the checkpoint.
        if model_value is None:
            child = CACHE.get('model_config', {}).get('backbone', {})
            if isinstance(child, Mapping):
                model_value = child.get(key, None)
        if model_value is not None:
            checks.append((f'model_structure:checkpoint_{key}', model_value == expected,
                           f'checkpoint model {key}={model_value!r}, required {expected!r}'))
    # Unsupported values must raise ValueError, not silently pick another
    # normalization/activation/FFN. Probe only clearly invalid strings;
    # lmax > 2 is not assumed invalid because some implementations extend it.
    for option in ('norm_type', 'act_type', 'ff_type'):
        try:
            cls = type(model)
            sig = inspect.signature(cls)
            conf = dict(CACHE['model_config'])
            if option in conf:
                conf[option] = '__definitely_unsupported_uma_option__'
            elif isinstance(conf.get('backbone'), Mapping) and option in conf['backbone']:
                conf['backbone'] = {**conf['backbone'], option: '__definitely_unsupported_uma_option__'}
            else:
                checks.append((f'model_structure:unsupported_{option}', True,
                               f'{option} not exposed in checkpoint model_config; constructor negative probe skipped'))
                continue
            accepts_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            kwargs = conf if accepts_var else {k: v for k, v in conf.items() if k in sig.parameters}
            if not kwargs and 'model_config' in sig.parameters:
                kwargs = {'model_config': conf}
            try:
                invalid_model = cls(**kwargs)
            except ValueError:
                checks.append((f'model_structure:unsupported_{option}', True,
                               f'unsupported {option} explicitly raises ValueError'))
            except Exception as exc:
                checks.append((f'model_structure:unsupported_{option}', False,
                               f'expected ValueError, got {type(exc).__name__}: {exc}'))
            else:
                del invalid_model
                checks.append((f'model_structure:unsupported_{option}', False,
                               f'invalid {option} was accepted by constructor'))
        except Exception:
            checks.append((f'model_structure:unsupported_{option}', False, traceback.format_exc()))
    return checks


def check_model_forward_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Mapping/object, force-gradient, periodic-input and unsupported-config probes."""
    if 'model' not in CACHE:
        loaded = check_checkpoint_loading({'dtype': 'float32'})
        if any(not passed for _, passed, _ in loaded):
            return [('model_forward:loaded_model', False, 'checkpoint load failed: ' + repr(loaded))]
    model = CACHE['model']
    checks = []
    dataset = config.get('dataset', 'omat')
    if isinstance(dataset, (list, tuple)):
        dataset = dataset[0]
    dataset = str(dataset)
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    pos = torch.tensor([[0., 0., 0.], [.9572, 0., 0.], [-.2390, .9270, 0.]],
                       dtype=dtype, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    data = {
        'atomic_numbers': torch.tensor([8, 1, 1], dtype=torch.long),
        'pos': pos, 'batch': torch.zeros(3, dtype=torch.long),
        'natoms': torch.tensor([3]), 'cell': torch.eye(3, dtype=dtype)[None] * 10.,
        'pbc': torch.zeros((1, 3), dtype=torch.bool),
        'charge': torch.zeros(1, dtype=dtype), 'spin': torch.zeros(1, dtype=dtype),
        'dataset': [dataset], 'edge_index': edge_index,
        'cell_offsets': torch.zeros((edge_index.shape[1], 3), dtype=dtype),
    }
    try:
        with torch.enable_grad():
            output = model(data)
        if not isinstance(output, Mapping):
            raise TypeError('UMA.forward must return dataset-prefixed mapping')
        flattened = {}
        pending = [('', output)]
        while pending:
            prefix, value = pending.pop()
            if isinstance(value, Mapping):
                pending.extend((f'{prefix}.{key}' if prefix else str(key), item)
                               for key, item in value.items())
            else:
                flattened[prefix] = value
        selected = {}
        for kind, aliases in {'energy': ('energy',), 'force': ('force', 'forces'),
                              'stress': ('stress', 'stresses')}.items():
            candidates = [(k, v) for k, v in flattened.items()
                          if dataset.lower() in k.lower() and any(k.lower().endswith(x) for x in aliases)]
            if not candidates:
                raise KeyError(f'missing dataset-prefixed {kind}: {list(flattened)}')
            selected[kind] = candidates[0][1]
        if not isinstance(selected['energy'], torch.Tensor) or selected['energy'].numel() != 1:
            raise AssertionError('energy must be a scalar-valued tensor per system')
        if not isinstance(selected['force'], torch.Tensor) or selected['force'].shape != (3, 3):
            raise AssertionError('force must be [N,3]')
        if not isinstance(selected['stress'], torch.Tensor) or selected['stress'].numel() not in (6, 9):
            raise AssertionError('stress must contain 6 or 9 components')
        if not all(torch.isfinite(v).all() for v in selected.values()):
            raise AssertionError('nonfinite forward prediction')
        checks.append(('model_forward:dataset_efs', True,
                       'dataset-prefixed scalar energy, [3,3] forces, stress all finite'))
    except Exception:
        return [('model_forward:dataset_efs', False, traceback.format_exc())]
    try:
        if not selected['energy'].requires_grad:
            raise AssertionError('energy must support differentiation')
        grad = torch.autograd.grad(selected['energy'].sum(), pos, retain_graph=True)[0]
        torch.testing.assert_close(selected['force'], -grad, atol=2e-5, rtol=2e-5)
        checks.append(('model_forward:force_energy_gradient', True,
                       f'force=-dE/dpos, max_abs={float((selected["force"]+grad).abs().max())}'))
    except Exception as gradient_error:
        # Some correct EFS implementations consume the autograd graph while
        # computing forces. In that case compare the directional derivative
        # of the independently recomputed energy with the force projection.
        try:
            direction = torch.tensor([[.2,-.1,.3],[-.4,.2,.1],[.1,.3,-.2]], dtype=dtype)
            direction = direction / direction.norm()
            eps = 1e-2 if dtype == torch.float32 else 1e-4
            energies = []
            for sign in (-1., 1.):
                changed = dict(data)
                changed['pos'] = (pos.detach() + sign * eps * direction).requires_grad_(True)
                with torch.enable_grad():
                    shifted = model(changed)
                flat = {}
                pending = [('', shifted)]
                while pending:
                    prefix, value = pending.pop()
                    if isinstance(value, Mapping):
                        pending.extend((f'{prefix}.{key}' if prefix else str(key), item)
                                       for key, item in value.items())
                    else:
                        flat[prefix] = value
                energy = next(value for key, value in flat.items()
                              if dataset.lower() in key.lower() and key.lower().endswith('energy'))
                energies.append(energy.detach().sum())
            derivative = (energies[1] - energies[0]) / (2 * eps)
            force_projection = -(selected['force'].detach() * direction).sum()
            torch.testing.assert_close(derivative, force_projection, atol=5e-2, rtol=2e-2)
            checks.append(('model_forward:force_energy_gradient', True,
                           f'finite-difference directional check, error={abs(float(derivative-force_projection)):.6g}'))
        except Exception:
            checks.append(('model_forward:force_energy_gradient', False,
                           f'primary autograd error: {gradient_error}\n' + traceback.format_exc()))
    try:
        obj = SimpleNamespace(**{k: v for k, v in data.items()})
        with torch.enable_grad():
            candidate = model(obj)
        if not isinstance(candidate, Mapping):
            raise TypeError('AtomicData-like object forward must return mapping')
        checks.append(('model_forward:object_input', True, 'Mapping and attribute-object inputs accepted'))
    except Exception:
        checks.append(('model_forward:object_input', False, traceback.format_exc()))
    try:
        periodic = dict(data)
        periodic['pbc'] = torch.ones((1, 3), dtype=torch.bool)
        periodic['cell_offsets'] = torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],
                                                  [0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], dtype=dtype)
        with torch.enable_grad():
            result = model(periodic)
        if not isinstance(result, Mapping):
            raise TypeError('periodic forward must return mapping')
        checks.append(('model_forward:periodic_graph', True, 'explicit edge_index/cell_offsets and PBC accepted'))
    except Exception:
        checks.append(('model_forward:periodic_graph', False, traceback.format_exc()))
    return checks


def check_model_forward_equivalence(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Compare first-principles forward predictions to the official standard runner."""
    if 'model' not in CACHE:
        loaded = check_checkpoint_loading({'dtype': 'float32'})
        if any(not passed for _, passed, _ in loaded):
            return [('forward_equivalence:loaded_model', False, 'checkpoint load failed: ' + repr(loaded))]
    try:
        previous_uma = sys.modules.pop('uma', None)
        original_path = sys.path[:]
        try:
            sys.path.insert(0, str(REFERENCE_DIR))
            ref_infer = _load_reference_modules('uma_inference.py')
            ref_load = getattr(ref_infer, 'load_model')
            sig = inspect.signature(ref_load)
            ref_kwargs = {'checkpoint_path': str(CACHE['checkpoint_path'])}
            if 'checkpoint_path' not in sig.parameters:
                ref_kwargs = {'checkpoint': str(CACHE['checkpoint_path'])}
            if 'device' in sig.parameters:
                ref_kwargs['device'] = 'cpu'
            standard = ref_load(**ref_kwargs)
        finally:
            sys.path[:] = original_path
            sys.modules.pop('uma', None)
            if previous_uma is not None:
                sys.modules['uma'] = previous_uma
        if not isinstance(standard, torch.nn.Module):
            raise TypeError('standard load_model did not produce module')
        CACHE['reference_infer'] = ref_infer
        CACHE['reference_model'] = standard.cpu().eval()
        checks = [('forward_equivalence:reference_checkpoint', True,
                   'official standard and submitted UMA loaded from the same checkpoint')]
        # The check must execute numerical comparisons, not merely load a model.
        comparisons = check_evaluate_oracle({'dataset': config.get('dataset', 'omat'),
                                             'atol': config.get('atol', 1e-6),
                                             'rtol': config.get('rtol', 1e-5),
                                             'max_fixtures': 1,
                                             'max_datasets': 1})
        checks.extend((f'forward_equivalence:{name}', ok, detail)
                      for name, ok, detail in comparisons)
        return checks
    except Exception:
        return [('forward_equivalence:reference_checkpoint', False, traceback.format_exc())]


def check_infer_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Real CLI help smoke plus ASE predict with a supplied model (no reloading)."""
    checks = []
    if 'model' not in CACHE or 'infer' not in CACHE:
        loaded = check_checkpoint_loading({'dtype': 'float32'})
        if any(not passed for _, passed, _ in loaded):
            return [('infer_execution:loaded_model', False, 'checkpoint load failed: ' + repr(loaded))]
    try:
        path = CANDIDATE_DIR / 'uma_infer.py'
        env = dict(os.environ)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        proc = subprocess.run([sys.executable, str(path), '--help'], cwd=CANDIDATE_DIR,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              timeout=int(config.get('help_timeout', 60)), env=env)
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f'CLI --help failed ({proc.returncode}): {proc.stderr[-2500:]}')
        checks.append(('infer_execution:cli_help', True, 'importable executable CLI with --help'))
    except Exception:
        checks.append(('infer_execution:cli_help', False, traceback.format_exc()))
    try:
        from ase import Atoms
        model = CACHE['model']
        mod = CACHE['infer']
        dataset = config.get('dataset', 'omat')
        if isinstance(dataset, (list, tuple)):
            dataset = dataset[0]
        atoms = Atoms(numbers=[8, 1, 1], positions=[[0, 0, 0], [.9572, 0, 0], [-.239, .927, 0]],
                      cell=[10, 10, 10], pbc=False)
        atoms.info.update(dataset=str(dataset), charge=0, spin=0)
        predict = getattr(mod, 'predict')
        signature = inspect.signature(predict)
        kwargs = {'model': model}
        for name in ('sample', 'samples', 'data', 'inputs', 'atoms', 'input_data', 'source'):
            if name in signature.parameters:
                kwargs[name] = atoms
                break
        else:
            raise TypeError('predict must accept a sample parameter')
        if 'dataset' in signature.parameters:
            kwargs['dataset'] = str(dataset)
        if 'device' in signature.parameters:
            kwargs['device'] = 'cpu'
        with mock.patch.object(mod, 'load_model', side_effect=AssertionError('predict reloaded checkpoint')):
            result = predict(**kwargs)
        CACHE['prediction'] = result
        checks.append(('infer_execution:predict_supplied_model', True,
                       f'ASE Atoms inference without duplicate model loading; return type={type(result).__name__}'))
        # Batch support is part of the prompt and should be exercised rather
        # than inferred from a source-code mention of "batch".
        batched_kwargs = dict(kwargs)
        batched_kwargs[next(k for k in ('sample','samples','data','inputs','atoms','input_data','source')
                            if k in batched_kwargs)] = [atoms.copy(), atoms.copy()]
        if 'batch_size' in signature.parameters:
            batched_kwargs['batch_size'] = 2
        with mock.patch.object(mod, 'load_model', side_effect=AssertionError('predict reloaded checkpoint')):
            batched_result = predict(**batched_kwargs)
        if not isinstance(batched_result, (Mapping, tuple, list)):
            raise TypeError('batch prediction must be a collection or structured outputs')
        checks.append(('infer_execution:batch_two', True,
                       'two ASE systems accepted with an existing checkpoint model'))
    except Exception:
        if not any(n == 'infer_execution:predict_supplied_model' for n, _, _ in checks):
            checks.append(('infer_execution:predict_supplied_model', False, traceback.format_exc()))
        checks.append(('infer_execution:batch_two', False, traceback.format_exc()))
    return checks


def check_infer_output_schema(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Require detached CPU numpy energy/forces/stress, rejecting tensors and NaNs."""
    if 'prediction' not in CACHE:
        execution = check_infer_execution({'dataset': config.get('dataset', 'omat')})
        if any(not passed for _, passed, _ in execution):
            return [('infer_output_schema:predict_result', False, 'inference failed: ' + repr(execution))]
    try:
        output = CACHE['prediction']
        if isinstance(output, (list, tuple)) and len(output) >= 2 and not isinstance(output[0], Mapping):
            output = {'energy': output[0], 'force': output[1],
                      **({'stress': output[2]} if len(output) > 2 else {})}
        if not isinstance(output, Mapping):
            raise TypeError('predict must return a mapping/tuple of numpy results')
        flat = {}
        pending = [('', output)]
        while pending:
            prefix, value = pending.pop()
            if isinstance(value, Mapping):
                pending.extend((f'{prefix}.{key}' if prefix else str(key), item)
                               for key, item in value.items())
            else:
                flat[prefix] = value
        picks = {}
        for key, aliases in {'energy': ('energy',), 'force': ('force', 'forces'),
                             'stress': ('stress', 'stresses')}.items():
            found = next((val for k, val in flat.items()
                          if any(k.lower().endswith(alias) for alias in aliases)), None)
            if found is None:
                raise KeyError(f'missing {key} in output: {list(flat)}')
            if not isinstance(found, np.ndarray):
                raise TypeError(f'{key} must be detached CPU numpy.ndarray, got {type(found)}')
            if not np.isfinite(found).all():
                raise AssertionError(f'{key} contains nonfinite values')
            picks[key] = found
        if picks['energy'].size != 1 or picks['force'].shape != (3, 3):
            raise AssertionError(f'energy/force shapes {picks["energy"].shape}, {picks["force"].shape}')
        if picks['stress'].size not in (6, 9):
            raise AssertionError(f'stress shape {picks["stress"].shape}')
        CACHE['normalized_prediction'] = picks
        return [('infer_output_schema:numpy_efs', True,
                 'finite CPU numpy energy scalar, forces [3,3], stress [6 or 9]')]
    except Exception:
        return [('infer_output_schema:numpy_efs', False, traceback.format_exc())]


def check_evaluate_oracle(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Compare energy/force predictions on real ASE structures with standard UMA.

    Per the old evaluator's contract the independent oracle is the standard
    repository implementation, not a labelled UMA test dataset.
    """
    checks = []
    if any(name not in CACHE for name in ('model', 'infer', 'reference_model', 'reference_infer')):
        equivalence = check_model_forward_equivalence({
            'dataset': config.get('dataset', 'omat'),
            'atol': config.get('atol', 1e-6), 'rtol': config.get('rtol', 1e-5),
        })
        if any(not passed for _, passed, _ in equivalence):
            return [('infer_oracle:setup', False, 'independent setup failed: ' + repr(equivalence))]
    try:
        from ase import Atoms
        candidate_infer, reference_infer = CACHE['infer'], CACHE['reference_infer']
        candidate_model, reference_model = CACHE['model'], CACHE['reference_model']
        dataset = config.get('dataset', 'omat')
        if isinstance(dataset, (list, tuple)):
            dataset = dataset[0]
        dataset = str(dataset)
        # Use the reference model's actual dataset keys when available.
        backbone = getattr(reference_model, 'backbone', reference_model)
        emb = getattr(backbone, 'dataset_embedding', None)
        dataset_dict = getattr(emb, 'dataset_emb_dict', None)
        if dataset_dict is not None and dataset not in dataset_dict:
            dataset = next(iter(dataset_dict))
        fixtures = [
            Atoms(numbers=[8,1,1,29], positions=[[0,0,0],[.9572,0,0],[-.239,.927,0],[2.4,.1,0]],
                  cell=[10,10,10], pbc=False),
            Atoms(numbers=[6,8,1,1], positions=[[0,0,0],[1.2,0,0],[-.5,1.0,0],[.1,-.4,1.0]],
                  cell=[7,7,7], pbc=True),
        ]
        datasets = [dataset]
        if dataset_dict is not None:
            for candidate in ('omat', 'omol', 'oc20', 'odac', 'omc'):
                if candidate in dataset_dict and candidate not in datasets:
                    datasets.append(candidate)
        datasets = datasets[:int(config.get('max_datasets', 2))]
        fixtures = fixtures[:int(config.get('max_fixtures', 2))]
        scenarios = []
        for selected_dataset in datasets:
            for atoms in fixtures:
                a = atoms.copy()
                a.info.update(dataset=selected_dataset, charge=0, spin=0)
                scenarios.append((selected_dataset, a))
        checks.append(('infer_oracle:fixtures', True,
                       f'{len(scenarios)} independent systems; datasets={datasets}; periodic/nonperiodic'))
    except Exception:
        return [('infer_oracle:fixtures', False, traceback.format_exc())]

    for index, (dataset, atoms) in enumerate(scenarios):
        try:
            outputs = {}
            for label, module, model in (('candidate', candidate_infer, candidate_model),
                                         ('reference', reference_infer, reference_model)):
                predict = getattr(module, 'predict')
                sig = inspect.signature(predict)
                kwargs = {'model': model}
                target = next((name for name in ('sample','samples','data','inputs','atoms','input_data','source')
                               if name in sig.parameters), None)
                if target is None:
                    raise TypeError(f'{label} predict lacks sample argument')
                kwargs[target] = atoms.copy()
                if 'device' in sig.parameters:
                    kwargs['device'] = 'cpu'
                if 'dataset' in sig.parameters:
                    kwargs['dataset'] = dataset
                # Use the standard's public predict interface without changing its internals.
                raw = predict(**kwargs)
                if isinstance(raw, (tuple, list)) and len(raw) >= 2 and not isinstance(raw[0], Mapping):
                    raw = {'energy': raw[0], 'force': raw[1],
                           **({'stress': raw[2]} if len(raw) > 2 else {})}
                if not isinstance(raw, Mapping):
                    raise TypeError(f'{label} predict must return mapping')
                flat = {}
                pending = [('', raw)]
                while pending:
                    prefix, value = pending.pop()
                    if isinstance(value, Mapping):
                        pending.extend((f'{prefix}.{key}' if prefix else str(key), item)
                                       for key, item in value.items())
                    else:
                        flat[prefix] = value
                results = {}
                for field, aliases in (('energy', ('energy',)), ('force', ('force','forces')),
                                       ('stress', ('stress','stresses'))):
                    value = next((v for k, v in flat.items() if any(k.lower().endswith(a) for a in aliases)), None)
                    if value is None:
                        if field == 'stress':
                            continue
                        raise KeyError(f'{label} missing {field}')
                    if torch.is_tensor(value):
                        raise TypeError(f'{label} returned tensor instead of numpy in predict')
                    results[field] = np.asarray(value, dtype=np.float64)
                outputs[label] = results
            for field in ('energy', 'force', 'stress'):
                if field not in outputs['reference'] or field not in outputs['candidate']:
                    if field == 'stress':
                        continue  # Older standard oracle only exposes energy/forces.
                    raise AssertionError(f'{field} missing from candidate/reference')
                actual, expected = outputs['candidate'][field], outputs['reference'][field]
                if actual.size != expected.size:
                    raise AssertionError(f'{field} size mismatch')
                a, b = actual.reshape(-1), expected.reshape(-1)
                if not np.allclose(a, b, atol=float(config.get('atol', 1e-6)),
                                   rtol=float(config.get('rtol', 1e-5))):
                    raise AssertionError(f'{field} mismatch max_abs={np.max(np.abs(a-b))}')
                checks.append((f'infer_oracle:frame{index}_{field}', True,
                               f'max_abs={float(np.max(np.abs(a-b))):.7g}'))
        except Exception:
            checks.append((f'infer_oracle:frame{index}', False, traceback.format_exc()))
    return checks
