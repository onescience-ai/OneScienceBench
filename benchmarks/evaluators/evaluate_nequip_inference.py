#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NequIP Cu inference: reusable base checks plus prompt-specific scientific checks.
Every task-specific check takes ``config`` and returns (name, passed, detail).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import math
import json
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml

from base.base_evaluator import (
    CANDIDATE_DIR,
    REFERENCE_DIR,
    _load_candidate_modules,
    _load_reference_modules,
    check_files,
    check_config as check_config_content,
    check_interface,
    check_infer_structure as _base_check_infer_structure,
    check_infer_output_schema as _base_check_infer_output_schema,
)
from base.model_structure_checker import check_model_structure_v2, checkpoints_to_tuples

CHECKPOINT = Path(
    '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/outputs/'
    'nequip_smoke_8dcu_20260817_171502/checkpoints/best.ckpt'
)
FIXED_DATA = Path(
    '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/'
    'demo/reference_data/smoke.xyz'
)
# Resolve once at module import, so every independent rule_eval check sees
# the same local fixture, not a global modified by another check.
LOCAL_DATA = CANDIDATE_DIR / 'smoke.xyz'
DATA = LOCAL_DATA if LOCAL_DATA.is_file() else FIXED_DATA
CHECKPOINT_SHA256 = '3ab4d7d891a36ed6346e661003fc30197d4b615171ccaec92cdd6b7bb1b4049b'
DATA_SHA256 = '6671f9b28d37484379f51f1deb23a6f4b165ed628467ee6bffa0648b6d50a5b5'
MODEL_PREFIX = 'model.sole_model.'
MODULE_ORDER = (
    'type_embed', 'spharm', 'edge_norm', 'bessel_encode', 'factor',
    'layer0_convnet', 'layer1_convnet', 'per_atom_energy_readout',
    'per_type_energy_scale_shift', 'total_energy_sum',
)
PREDICTION_KEYS = ('energy', 'forces', 'stress')
RECORD_KEYS = {
    'energy': ('energy', 'energy_ev'),
    'forces': ('forces', 'forces_ev_per_angstrom'),
    'stress': ('stress', 'stress_ev_per_angstrom3'),
}
# The fixed paths define the task contract. The active path may change only
# to a byte-identical SHA256-verified copy when a shared fixture is unreadable.
FIXED_CHECKPOINT = CHECKPOINT



def check_infer_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Reuse base AST checks; enforce an independent local NequIP implementation."""
    checks = []
    try:
        shared = _base_check_infer_structure(
            config, caller_globals=caller_globals, caller_locals=caller_locals
        )
        # Neither a literal .eval() in the inspected files nor no_grad is a
        # valid requirement for gradient-derived stress/force. Eval behaviour
        # is tested dynamically in the forward / CLI checks.
        excluded = {'inference_start:has_eval', 'inference_start:has_no_grad'}
        checks.extend((n, ok, d) for n, ok, d in shared if n not in excluded)
    except Exception:
        checks.append(('infer_structure:base_ast', False, traceback.format_exc()))

    try:
        package = CANDIDATE_DIR / config.get('portable_package', 'portable_nequip')
        files = sorted(package.rglob('*.py'))
        if not package.is_dir() or not (package / '__init__.py').is_file() or not files:
            raise AssertionError('portable_nequip must be a local Python package')
        if any(not f.resolve().is_relative_to(CANDIDATE_DIR.resolve()) for f in files):
            raise AssertionError('package source must not escape submission directory')
        if not any(f.read_text(encoding='utf-8').strip() for f in files):
            raise AssertionError('local package contains no implementation')
        checks.append(('infer_structure:portable_package', True, f'{len(files)} source files'))
    except Exception:
        checks.append(('infer_structure:portable_package', False, traceback.format_exc()))

    try:
        forbidden = set(config.get('prohibited_imports', ['onescience', 'nequip', 'mace', 'mattersim']))
        violations = []
        root = CANDIDATE_DIR.resolve()
        for path in CANDIDATE_DIR.rglob('*.py'):
            if not path.resolve().is_relative_to(root):
                raise AssertionError(f'source escapes candidate directory: {path}')
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str) and (
                        isinstance(node.func, ast.Name) and node.func.id == '__import__'
                        or isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'
                    ):
                        modules = [node.args[0].value]
                for mod in modules:
                    if mod.split('.')[0] in forbidden:
                        violations.append(f'{path.name}:{getattr(node, "lineno", 0)} {mod}')
                if isinstance(node, ast.Call):
                    callee = (node.func.id if isinstance(node.func, ast.Name) else
                              node.func.attr if isinstance(node.func, ast.Attribute) else '')
                    if callee in {'open', 'Path', 'read_text', 'read_bytes', 'load'}:
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                literal = arg.value.replace('\\', '/')
                                if any(marker in literal for marker in (
                                    'benchmarks/standard/', 'benchmarks/evaluators/',
                                    'evaluation/eval_datasets/codegen/standard/',
                                    'evaluation/eval_datasets/codegen/evaluators/',
                                )):
                                    violations.append(f'{path.name}:{getattr(node, "lineno", 0)} prohibited file access')
        if violations:
            raise AssertionError('; '.join(violations[:15]))
        checks.append(('infer_structure:independent_source', True, 'no forbidden imports/reference access'))
    except Exception:
        checks.append(('infer_structure:independent_source', False, traceback.format_exc()))
    return checks


def check_data_availability(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Verify the fixed smoke fixture's SHA256, using a local byte-identical copy."""
    name = 'data_availability:smoke_xyz_sha256'
    try:
        # data_file in the JSON still records the canonical dataset identity.
        requested = Path(config.get('data_file', FIXED_DATA))
        expected = str(config.get('data_sha256', DATA_SHA256))
        if requested != FIXED_DATA or expected != DATA_SHA256:
            raise AssertionError('data path / SHA contract differs from the original prompt')

        digest = hashlib.sha256()
        with DATA.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != DATA_SHA256:
            raise AssertionError(f'smoke.xyz SHA256 mismatch at {DATA}: {actual}')

        source = 'candidate copy' if DATA == LOCAL_DATA else 'canonical dataset'
        return [(name, True, f'{source}: SHA256 verified at {DATA}')]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_checkpoint_loading(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Independently verify complete raw state, ordered EMA and live model state."""
    global CHECKPOINT
    results = []
    try:
        requested = Path(config.get('checkpoint_file', FIXED_CHECKPOINT))
        expected_hash = config.get('checkpoint_sha256', CHECKPOINT_SHA256)
        if requested != FIXED_CHECKPOINT or expected_hash != CHECKPOINT_SHA256:
            raise AssertionError('checkpoint contract differs from the prompt')
        errors = []
        for candidate in (FIXED_CHECKPOINT, CANDIDATE_DIR / 'best.ckpt'):
            try:
                digest = hashlib.sha256()
                with candidate.open('rb') as f:
                    for chunk in iter(lambda: f.read(4 << 20), b''):
                        digest.update(chunk)
            except OSError as exc:
                errors.append(f'{candidate}: {type(exc).__name__}: {exc}')
                continue
            if digest.hexdigest() != CHECKPOINT_SHA256:
                raise AssertionError(f'checkpoint SHA256 mismatch at {candidate}')
            CHECKPOINT = candidate
            break
        else:
            raise OSError('no accessible valid checkpoint; ' + '; '.join(errors))
        results.append(('checkpoint_loading:sha256', True, str(CHECKPOINT)))
    except Exception:
        return [('checkpoint_loading:sha256', False, traceback.format_exc())]

    try:
        safe, strict = False, False
        for path in CANDIDATE_DIR.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                if node.func.attr == 'load' and isinstance(kw.get('weights_only'), ast.Constant):
                    safe |= kw['weights_only'].value is True
                if node.func.attr == 'load_state_dict' and isinstance(kw.get('strict'), ast.Constant):
                    strict |= kw['strict'].value is True
        if not (safe and strict):
            raise AssertionError(f'require explicit torch.load(weights_only=True) and strict=True; got {safe}, {strict}')
        results.append(('checkpoint_loading:source_safe_strict', True, 'safe load and strict state restore'))
    except Exception:
        results.append(('checkpoint_loading:source_safe_strict', False, traceback.format_exc()))

    try:
        payload = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get('state_dict'), Mapping):
            raise AssertionError('expected a Lightning checkpoint with state_dict')
        state_dict = payload['state_dict']
        prefix = config.get('ema_source_prefix', MODEL_PREFIX)
        nraw = int(config.get('raw_state_count', 33))
        numel = int(config.get('raw_state_numel', 1855))
        nstate = int(config.get('model_state_count', 33))
        nema = int(config.get('ema_tensor_count', 11))
        if (prefix, nraw, numel, nstate, nema) != (MODEL_PREFIX, 33, 1855, 33, 11):
            raise AssertionError('unexpected state/EMA quantity contract')
        raw = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if len(raw) != nraw or any(not isinstance(x, torch.Tensor) for x in raw.values()):
            raise AssertionError(f'raw state must have {nraw} tensor entries; got {len(raw)}')
        if sum(t.numel() for t in raw.values()) != numel:
            raise AssertionError('raw checkpoint element count differs from 1855')
        results.append(('checkpoint_loading:raw_state', True, '33 tensors / 1855 elements'))
    except Exception:
        return results + [('checkpoint_loading:raw_state', False, traceback.format_exc())]

    try:
        mod = _load_candidate_modules('nequip_inference.py')
        imported = mod.load_checkpoint_payload(str(CHECKPOINT))
        if not isinstance(imported, Mapping):
            raise TypeError('load_checkpoint_payload must return checkpoint mapping')
        supplied_raw = mod.raw_model_state(imported)
        if not isinstance(supplied_raw, Mapping) or set(supplied_raw) != set(raw):
            raise AssertionError('raw_model_state did not restore the complete model.sole_model. state')
        for key, value in raw.items():
            if not isinstance(supplied_raw[key], torch.Tensor) or not torch.equal(value.cpu(), supplied_raw[key].cpu()):
                raise AssertionError(f'incorrect raw model state tensor {key}')
        results.append(('checkpoint_loading:candidate_raw_extraction', True, 'all raw tensors match'))
    except Exception:
        return results + [('checkpoint_loading:candidate_raw_extraction', False, traceback.format_exc())]

    try:
        conf = yaml.safe_load((CANDIDATE_DIR / 'nequip_inference.yaml').read_text(encoding='utf-8'))
        if not isinstance(conf, dict):
            raise TypeError('YAML config must be a dictionary')
        conf = conf.get('model', conf)
        signature = inspect.signature(mod.load_model)
        params = signature.parameters
        kwargs = {}
        for key in ('config', 'model_config'):
            if key in params:
                kwargs[key] = conf
                break
        for key in ('checkpoint_path', 'checkpoint', 'ckpt_path', 'path'):
            if key in params:
                kwargs[key] = str(CHECKPOINT)
                break
        if 'device' in params:
            kwargs['device'] = 'cpu'
        if not any(key in kwargs for key in ('checkpoint_path', 'checkpoint', 'ckpt_path', 'path')):
            # Typical standard interface: load_model(checkpoint_path, device='cpu')
            loaded = mod.load_model(str(CHECKPOINT), 'cpu')
        else:
            loaded = mod.load_model(**kwargs)
        model = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
        if not isinstance(model, torch.nn.Module):
            raise TypeError('load_model must produce torch.nn.Module (or tuple with model first)')
        current = model.state_dict()
        if len(current) != nstate or set(current) != set(raw):
            raise AssertionError('candidate model state is missing keys or contains extra keys')
        if any(current[k].shape != raw[k].shape or current[k].dtype != raw[k].dtype for k in raw):
            raise AssertionError('candidate model state metadata differs from checkpoint')
        param_names = tuple(name for name, _ in model.named_parameters())
        if len(param_names) != nema or not set(param_names).issubset(raw):
            raise AssertionError(f'expect 11 model parameters restored by ordered EMA, got {len(param_names)}')
        results.append(('checkpoint_loading:complete_model_state', True, '33 matched state keys; 11 ordered parameters'))
    except Exception:
        return results + [('checkpoint_loading:complete_model_state', False, traceback.format_exc())]

    try:
        eval_signature = inspect.signature(mod.evaluation_model_state)
        eval_kwargs = {}
        if 'model' in eval_signature.parameters:
            claimed_eval = mod.evaluation_model_state(imported, model=model)
        else:
            claimed_eval = mod.evaluation_model_state(imported)
        if not isinstance(claimed_eval, Mapping) or set(claimed_eval) != set(raw):
            raise AssertionError('evaluation_model_state must preserve all 33 state entries')
        if any(not torch.equal(current[k].cpu(), claimed_eval[k].cpu()) for k in raw):
            raise AssertionError('live model state differs from evaluation_model_state')
        if any(not torch.equal(raw[k], current[k].cpu()) for k in raw if k not in param_names):
            raise AssertionError('EMA unexpectedly changed a non-parameter buffer')
        if not any(not torch.equal(raw[k], current[k].cpu()) for k in param_names):
            raise AssertionError('11 EMA parameters were not actually substituted')

        # Reference is an independent, trusted oracle for EMA-by-parameter-order.
        for key in tuple(sys.modules):
            if key == 'portable_nequip' or key.startswith('portable_nequip.'):
                sys.modules.pop(key, None)
        reference = str(REFERENCE_DIR)
        sys.path[:] = [reference] + [entry for entry in sys.path if entry != reference]
        std = _load_reference_modules('nequip_inference.py')
        reference_loaded = std.load_model(str(CHECKPOINT), 'cpu')
        ref_model = reference_loaded[0] if isinstance(reference_loaded, (list, tuple)) else reference_loaded
        ref_state = ref_model.state_dict()
        if set(ref_state) != set(raw) or any(
            not torch.equal(ref_state[k].cpu(), current[k].cpu()) for k in raw
        ):
            raise AssertionError('EMA state does not equal independently loaded reference')
        strict_result = model.load_state_dict(claimed_eval, strict=True)
        if strict_result.missing_keys or strict_result.unexpected_keys:
            raise AssertionError('strict load reported missing or unexpected keys')
        results.append(('checkpoint_loading:ema_strict_reference', True,
                        '11 EMA tensors, 33 exact values and strict=True match reference'))
    except Exception:
        results.append(('checkpoint_loading:ema_strict_reference', False, traceback.format_exc()))
    finally:
        for key in tuple(sys.modules):
            if key == 'portable_nequip' or key.startswith('portable_nequip.'):
                sys.modules.pop(key, None)
        candidate = str(CANDIDATE_DIR)
        sys.path[:] = [candidate] + [entry for entry in sys.path if entry != candidate]
    return results


def check_model_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check registered stages; allow scalar factor and inlined final sum."""
    try:
        mod = _load_candidate_modules('nequip_inference.py')
        conf = yaml.safe_load((CANDIDATE_DIR / 'nequip_inference.yaml').read_text(encoding='utf-8'))
        conf = conf.get('model', conf)
        sig = inspect.signature(mod.load_model)
        args = {}
        if 'config' in sig.parameters: args['config'] = conf
        if 'model_config' in sig.parameters: args['model_config'] = conf
        for name in ('checkpoint_path', 'checkpoint', 'ckpt_path', 'path'):
            if name in sig.parameters:
                args[name] = str(CHECKPOINT)
                break
        if 'device' in sig.parameters: args['device'] = 'cpu'
        obj = mod.load_model(**args) if any(n in args for n in
            ('checkpoint_path', 'checkpoint', 'ckpt_path', 'path')) else mod.load_model(str(CHECKPOINT), 'cpu')
        model = obj[0] if isinstance(obj, (list, tuple)) else obj
        if not isinstance(model, torch.nn.Module):
            raise TypeError('load_model did not return an nn.Module')
        checks = [('model_structure:build', True, type(model).__name__)]
    except Exception:
        return [('model_structure:build', False, traceback.format_exc())]

    try:
        rules = config.get('rules', [])
        if not rules:
            raise ValueError('missing structure rules')
        checks.extend(checkpoints_to_tuples(check_model_structure_v2(
            model, config_dict={'rules': rules}, verbose=False
        )))
    except Exception:
        checks.append(('model_structure:declarative_rules', False, traceback.format_exc()))

    try:
        expected = tuple(config.get('network_module_order', MODULE_ORDER))
        if expected != MODULE_ORDER or int(config.get('layer_count', 2)) != 2:
            raise AssertionError('unrecognized NequIP layer ordering contract')
        # factor may be a scalar coefficient; total_energy_sum can be a
        # functional reduction. They need not be registered nn.Module objects.
        registered = tuple(n for n in expected if n not in ('factor', 'total_energy_sum'))
        matches = []
        for module in model.modules():
            names = tuple(module._modules)
            if all(name in names for name in registered):
                positions = [names.index(name) for name in registered]
                if positions == sorted(positions):
                    matches.append(module)
        if len(matches) != 1:
            raise AssertionError(f'expected one correctly ordered NequIP network; got {len(matches)}')
        seq = matches[0]
        if sum(type(child).__name__ == 'ConvNetLayer' for child in seq.modules()) != 2:
            raise AssertionError('model must contain exactly two ConvNetLayer stages')
        factor = getattr(seq, 'factor', None)
        if factor is not None and not isinstance(factor, torch.nn.Module):
            f = torch.as_tensor(factor)
            if f.numel() != 1 or not bool(torch.isfinite(f).all()):
                raise AssertionError('functional factor must be a finite scalar')
        checks.append(('model_structure:ordered_stages', True,
                       'eight registered stages in order; functional factor and sum accepted'))
    except Exception:
        checks.append(('model_structure:ordered_stages', False, traceback.format_exc()))
    return checks


def check_model_forward_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Run actual periodic ASE inference with a legitimate candidate signature."""
    names = ('model_forward:periodic_ase', 'model_forward:efs_schema')
    try:
        from ase.io import read
        atoms = read(str(DATA), index=int(config.get('structure_index', 0)))
        if not len(atoms) or not bool(np.asarray(atoms.pbc).any()):
            raise AssertionError('expected a nonempty periodic ASE structure')
        mod = _load_candidate_modules('nequip_inference.py')
        conf = yaml.safe_load((CANDIDATE_DIR / 'nequip_inference.yaml').read_text(encoding='utf-8'))
        conf = conf.get('model', conf)
        sig = inspect.signature(mod.load_model)
        args = {}
        if 'config' in sig.parameters: args['config'] = conf
        if 'model_config' in sig.parameters: args['model_config'] = conf
        for name in ('checkpoint_path', 'checkpoint', 'ckpt_path', 'path'):
            if name in sig.parameters:
                args[name] = str(CHECKPOINT)
                break
        if 'device' in sig.parameters: args['device'] = 'cpu'
        loaded = mod.load_model(**args) if any(n in args for n in
            ('checkpoint_path', 'checkpoint', 'ckpt_path', 'path')) else mod.load_model(str(CHECKPOINT), 'cpu')
        model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        model.eval()
        checks = [(names[0], True, f'ASE Atoms N={len(atoms)}')]
    except Exception:
        return [(n, False, traceback.format_exc()) for n in names]
    try:
        sig = inspect.signature(mod.evaluate_structure)
        params = {}
        if 'r_max' in sig.parameters: params['r_max'] = float(conf.get('r_max', 4.0))
        if 'device' in sig.parameters: params['device'] = 'cpu'
        with torch.enable_grad():
            result = mod.evaluate_structure(model, atoms, **params)
        if not isinstance(result, Mapping) or not set(PREDICTION_KEYS).issubset(result):
            raise AssertionError('evaluate_structure must return energy, forces and stress')
        expected = {'energy': (1,), 'forces': (len(atoms), 3), 'stress': (1, 3, 3)}
        for name, shape in expected.items():
            tensor = result[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise AssertionError(f'{name} must have shape {shape}')
            if not torch.isfinite(tensor).all():
                raise AssertionError(f'{name} has nonfinite values')
        checks.append((names[1], True, 'energy, forces, stress shapes and values correct'))
    except Exception:
        checks.append((names[1], False, traceback.format_exc()))
    return checks


def check_model_forward_equivalence(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Compare multiple ASE configurations against independent EMA reference."""
    names = ('model_forward_equivalence:ema_live_state', 'model_forward_equivalence:energy_force_stress')
    try:
        from ase.io import read
        indices = [int(v) for v in config.get('sample_indices', [0, 1])]
        if not indices:
            raise AssertionError('no samples selected')
        structures = [read(str(DATA), index=i) for i in indices]
        for key in tuple(sys.modules):
            if key == 'portable_nequip' or key.startswith('portable_nequip.'):
                sys.modules.pop(key, None)
        ref_path = str(REFERENCE_DIR)
        sys.path[:] = [ref_path] + [p for p in sys.path if p != ref_path]
        reference = _load_reference_modules('nequip_inference.py')
        std_loaded = reference.load_model(str(CHECKPOINT), 'cpu')
        ref_model = std_loaded[0] if isinstance(std_loaded, (tuple, list)) else std_loaded
        ref_model.eval()
        ref_state = ref_model.state_dict()
        rsig = inspect.signature(reference.evaluate_structure)
        rargs = {}
        if 'r_max' in rsig.parameters: rargs['r_max'] = 4.0
        if 'device' in rsig.parameters: rargs['device'] = 'cpu'
        with torch.enable_grad():
            expected = [reference.evaluate_structure(ref_model, a.copy(), **rargs) for a in structures]

        for key in tuple(sys.modules):
            if key == 'portable_nequip' or key.startswith('portable_nequip.'):
                sys.modules.pop(key, None)
        cand_path = str(CANDIDATE_DIR)
        sys.path[:] = [cand_path] + [p for p in sys.path if p != cand_path]
        mod = _load_candidate_modules('nequip_inference.py')
        conf = yaml.safe_load((CANDIDATE_DIR / 'nequip_inference.yaml').read_text(encoding='utf-8'))
        conf = conf.get('model', conf)
        sig = inspect.signature(mod.load_model)
        kwargs = {}
        if 'config' in sig.parameters: kwargs['config'] = conf
        if 'model_config' in sig.parameters: kwargs['model_config'] = conf
        for k in ('checkpoint_path', 'checkpoint', 'ckpt_path', 'path'):
            if k in sig.parameters: kwargs[k] = str(CHECKPOINT); break
        if 'device' in sig.parameters: kwargs['device'] = 'cpu'
        loaded = mod.load_model(**kwargs) if any(k in kwargs for k in
            ('checkpoint_path', 'checkpoint', 'ckpt_path', 'path')) else mod.load_model(str(CHECKPOINT), 'cpu')
        model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        model.eval()
        csig = inspect.signature(mod.evaluate_structure)
        cargs = {}
        if 'r_max' in csig.parameters: cargs['r_max'] = float(conf.get('r_max', 4.0))
        if 'device' in csig.parameters: cargs['device'] = 'cpu'
        with torch.enable_grad():
            actual = [mod.evaluate_structure(model, a.copy(), **cargs) for a in structures]
    except Exception:
        return [(n, False, traceback.format_exc()) for n in names]
    finally:
        for key in tuple(sys.modules):
            if key == 'portable_nequip' or key.startswith('portable_nequip.'):
                sys.modules.pop(key, None)
        path = str(CANDIDATE_DIR)
        sys.path[:] = [path] + [p for p in sys.path if p != path]

    try:
        live = model.state_dict()
        if len(ref_state) != 33 or set(live) != set(ref_state):
            raise AssertionError('candidate and reference must share all 33 state keys')
        for key, value in ref_state.items():
            if not torch.equal(live[key].cpu(), value.cpu()):
                raise AssertionError(f'model EMA differs at state tensor {key}')
        checks = [(names[0], True, 'all 33 parameter/buffer tensors identical to reference')]
    except Exception:
        checks = [(names[0], False, traceback.format_exc())]
    try:
        max_err = {}
        atol, rtol = float(config.get('atol', 1e-6)), float(config.get('rtol', 1e-5))
        for gold, pred in zip(expected, actual):
            for key in PREDICTION_KEYS:
                torch.testing.assert_close(pred[key], gold[key], atol=atol, rtol=rtol)
                err = float((pred[key].double() - gold[key].double()).abs().max())
                max_err[key] = max(max_err.get(key, 0.0), err)
        checks.append((names[1], True, f'{len(actual)} real frames; max_abs={max_err}'))
    except Exception:
        checks.append((names[1], False, traceback.format_exc()))
    return checks


def check_infer_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Execute a fresh CLI run with SHA-verified dataset and checkpoint."""
    try:
        output = (CANDIDATE_DIR / config.get('output_path', 'nequip_predictions.json')).resolve()
        if not output.is_relative_to(CANDIDATE_DIR.resolve()):
            raise ValueError('output must stay inside submission directory')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        command = [
            sys.executable, str(CANDIDATE_DIR / 'run_inference.py'),
            '--config', str(CANDIDATE_DIR / 'nequip_inference.yaml'),
            '--checkpoint', str(CHECKPOINT), '--input', str(DATA),
            '--output', str(output), '--device', 'cpu',
        ]
        process = subprocess.run(command, cwd=str(CANDIDATE_DIR), text=True,
                                 capture_output=True, timeout=int(config.get('timeout', 600)))
        if process.returncode != 0:
            raise RuntimeError(f'CLI exited {process.returncode}; stderr={process.stderr[-3000:]}')
        if not output.is_file() or not output.stat().st_size:
            raise AssertionError('CLI did not create a fresh nonempty predictions file')
        data = json.loads(output.read_text(encoding='utf-8'))
        if not isinstance(data, (list, dict)) or not data:
            raise ValueError('CLI output is empty or not JSON records')
        return [('infer_execution:fresh_cli', True, str(output))]
    except Exception:
        return [('infer_execution:fresh_cli', False, traceback.format_exc())]


def check_infer_output_schema(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Reuse the base JSON checks and validate equivalent E/F/S field aliases."""
    checks = list(_base_check_infer_output_schema(
        config, caller_globals=caller_globals, caller_locals=caller_locals
    ))
    try:
        value = json.loads((CANDIDATE_DIR / config.get('output_path', 'nequip_predictions.json')).read_text(encoding='utf-8'))
        if isinstance(value, list):
            records = value
        elif isinstance(value, Mapping):
            records = next((value[k] for k in ('predictions', 'results', 'structures', 'outputs', 'data')
                            if isinstance(value.get(k), list)), None)
            if records is None and all(any(alias in value for alias in RECORD_KEYS[name]) for name in PREDICTION_KEYS):
                records = [value]
            if records is None and value and all(isinstance(item, Mapping) for item in value.values()):
                records = list(value.values())
        else:
            records = None
        n = int(config.get('expected_structures', 8))
        if not isinstance(records, list) or len(records) != n:
            raise AssertionError(f'expected {n} structures, got {len(records) if isinstance(records, list) else None}')
        checks.append(('infer_output_schema:structure_count', True, str(n)))
    except Exception:
        return checks + [('infer_output_schema:structure_count', False, traceback.format_exc()),
                         ('infer_output_schema:energy_force_stress', False, 'missing records')]
    try:
        from ase.io import read
        structures = read(str(DATA), index=':')
        if len(structures) != len(records):
            raise AssertionError('prediction record count differs from smoke.xyz')
        for i, (record, atoms) in enumerate(zip(records, structures)):
            if not isinstance(record, Mapping):
                raise AssertionError(f'record {i} must be a mapping')
            fields = {}
            for name in PREDICTION_KEYS:
                key = next((alias for alias in RECORD_KEYS[name] if alias in record), None)
                if key is None:
                    raise AssertionError(f'record {i} lacks {name}')
                fields[name] = np.asarray(record[key], dtype=np.float64)
            energy = fields['energy'].reshape(-1)
            if energy.shape != (1,) or fields['forces'].shape != (len(atoms), 3) or fields['stress'].shape not in ((3, 3), (1, 3, 3)):
                raise AssertionError(f'record {i} E/F/S shapes are invalid')
            if not all(np.isfinite(x).all() for x in fields.values()):
                raise AssertionError(f'record {i} has NaN or Inf')
        checks.append(('infer_output_schema:energy_force_stress', True, 'all structures finite and correct shape'))
    except Exception:
        checks.append(('infer_output_schema:energy_force_stress', False, traceback.format_exc()))
    return checks


def check_evaluate_oracle(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Independently compare fresh CLI predictions on the whole smoke fixture."""
    names = ('infer_oracle:complete_smoke', 'infer_oracle:energy',
             'infer_oracle:forces', 'infer_oracle:stress')
    try:
        from ase.io import read
        structures = read(str(DATA), index=':')
        if len(structures) != int(config.get('expected_structures', 8)):
            raise AssertionError('fixed smoke set size differs from contract')
        payload = json.loads((CANDIDATE_DIR / config.get('output_path', 'nequip_predictions.json')).read_text(encoding='utf-8'))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping):
            records = next((payload[k] for k in ('predictions', 'results', 'structures', 'outputs', 'data')
                            if isinstance(payload.get(k), list)), None)
            if records is None and payload and all(isinstance(x, Mapping) for x in payload.values()):
                records = list(payload.values())
        else:
            records = None
        if not isinstance(records, list) or len(records) != len(structures):
            raise AssertionError('not all original smoke frames were evaluated')
        for key in tuple(sys.modules):
            if key == 'portable_nequip' or key.startswith('portable_nequip.'):
                sys.modules.pop(key, None)
        ref_path = str(REFERENCE_DIR)
        sys.path[:] = [ref_path] + [p for p in sys.path if p != ref_path]
        std = _load_reference_modules('nequip_inference.py')
        loaded = std.load_model(str(CHECKPOINT), 'cpu')
        model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        model.eval()
        sig = inspect.signature(std.evaluate_structure)
        kwargs = {}
        if 'r_max' in sig.parameters: kwargs['r_max'] = 4.0
        if 'device' in sig.parameters: kwargs['device'] = 'cpu'
        with torch.enable_grad():
            expected = [std.evaluate_structure(model, a.copy(), **kwargs) for a in structures]
        checks = [(names[0], True, f'full set {len(structures)} independent reference inferences')]
    except Exception:
        return [(n, False, traceback.format_exc()) for n in names]
    finally:
        for key in tuple(sys.modules):
            if key == 'portable_nequip' or key.startswith('portable_nequip.'):
                sys.modules.pop(key, None)
        path = str(CANDIDATE_DIR)
        sys.path[:] = [path] + [p for p in sys.path if p != path]

    atol, rtol = float(config.get('atol', 1e-6)), float(config.get('rtol', 1e-5))
    for name in PREDICTION_KEYS:
        try:
            maximum = 0.0
            for i, (record, gold) in enumerate(zip(records, expected)):
                alias = next((key for key in RECORD_KEYS[name] if isinstance(record, Mapping) and key in record), None)
                if alias is None:
                    raise AssertionError(f'missing {name} at sample {i}')
                actual = torch.as_tensor(record[alias], dtype=torch.float64).reshape(-1)
                truth = gold[name].detach().cpu().double().reshape(-1)
                if actual.shape != truth.shape:
                    raise AssertionError(f'{name} shape differs at sample {i}')
                torch.testing.assert_close(actual, truth, atol=atol, rtol=rtol)
                maximum = max(maximum, float((actual - truth).abs().max()))
            checks.append((f'infer_oracle:{name}', True, f'max_abs={maximum:.8g}'))
        except Exception:
            checks.append((f'infer_oracle:{name}', False, traceback.format_exc()))
    return checks


