#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import inspect
import json
import math
import re
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from base.base_evaluator import (
    CANDIDATE_DIR,
    _load_candidate_modules,
    _load_reference_modules,
    check_files,
    check_interface,
    check_infer_structure,
    check_infer_output_schema as _base_check_infer_output_schema,
)
from base.model_structure_checker import check_model_structure_v2, checkpoints_to_tuples

CHECKPOINT = Path('/public/share/sugonhpcapp01/onestore/paper_repro/weight/rno/best_model.pth')
DATA = Path('/public/share/sugonhpcapp01/onestore/onedatasets/CFD_Benchmark/darcy/piececonst_r421_N1024_smooth2.mat')
EXPECTED = {
    'in_channels': 3, 'out_channels': 1, 'channels': 128,
    'layers': 8, 'heads': 8, 'slices': 32, 'angles': 32,
    'radon_indices': (1, 3, 5, 7), 'angle_hidden': 16, 'mlp_ratio': 4.0,
}


def check_data_availability(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Verify canonical readable MAT, coeff/sol shapes, and fixed selection protocol."""
    checks = []
    path = Path(config.get('data_file', DATA))
    try:
        if path.resolve() != DATA.resolve() or not path.is_file():
            raise FileNotFoundError(f'fixed Darcy MAT not found: {DATA}')
        with path.open('rb') as stream:
            if not stream.read(128):
                raise ValueError('MAT file is empty')
        checks.append(('data_availability:canonical_mat', True, 'canonical MAT readable'))
    except Exception:
        return [('data_availability:canonical_mat', False, traceback.format_exc())]
    try:
        import scipy.io as sio
        try:
            metadata = {name: tuple(shape) for name, shape, _ in sio.whosmat(str(path))}
        except (ValueError, NotImplementedError):
            import h5py
            with h5py.File(path, 'r') as data:
                metadata = {key: tuple(data[key].shape) for key in data if hasattr(data[key], 'shape')}
        for key in ('coeff', 'sol'):
            shape = metadata.get(key)
            if shape not in ((1024, 421, 421), (421, 421, 1024)):
                raise AssertionError(f'{key} MAT shape mismatch: {shape}')
        checks.append(('data_availability:mat_fields_and_shapes', True, 'coeff/sol each 1024x421x421'))
    except Exception:
        checks.append(('data_availability:mat_fields_and_shapes', False, traceback.format_exc()))
    try:
        count = int(config.get('samples', 200))
        stride = int(config.get('stride', 5))
        resolution = list(config.get('resolution', [85, 85]))
        if (count, stride, resolution) != (200, 5, [85, 85]):
            raise AssertionError('expected first 200, stride=5, spatial 85x85')
        if len(range(0, 421, stride)) != 85:
            raise AssertionError('incorrect spatial downsampling length')
        checks.append(('data_availability:selection_protocol', True, 'first 200, ::5, 85x85'))
    except Exception:
        checks.append(('data_availability:selection_protocol', False, traceback.format_exc()))
    return checks


def check_checkpoint_loading(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Validate real checkpoint tensors, normalizers, model construction and strict load."""
    names = ('checkpoint_loading:checkpoint_metadata',
             'checkpoint_loading:checkpoint_statistics',
             'checkpoint_loading:derived_architecture',
             'checkpoint_loading:strict_model_state',
             'checkpoint_loading:candidate_load_contract')
    try:
        location = Path(config.get('checkpoint_file', CHECKPOINT))
        if location.resolve() != CHECKPOINT.resolve():
            raise AssertionError('checkpoint path must be the fixed prompt asset')
        payload = torch.load(location, map_location='cpu', weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError('checkpoint must be a mapping')
        state = next((payload[k] for k in config.get('state_keys', ['model', 'model_state_dict', 'state_dict'])
                      if isinstance(payload.get(k), Mapping)), None)
        if not state or len(state) != 178 or any(not isinstance(v, torch.Tensor) for v in state.values()):
            raise AssertionError('checkpoint must contain exactly 178 model tensors')
        checks = [(names[0], True, '178 checkpoint state tensors present')]
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    try:
        for key in ('input_normalizer', 'output_normalizer'):
            entry = payload.get(key)
            if not isinstance(entry, Mapping) or not {'mean', 'std'} <= set(entry):
                raise KeyError(f'{key}: missing mean/std')
            mean, std = (torch.as_tensor(entry[stat]).float() for stat in ('mean', 'std'))
            eps = float(entry.get('eps', 1e-6))
            if tuple(mean.shape) != (1, 1, 85, 85) or std.shape != mean.shape:
                raise AssertionError(f'{key} statistic shape {mean.shape}/{std.shape}')
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any() or eps <= 0:
                raise AssertionError(f'{key} contains invalid values')
        checks.append((names[1], True, 'both checkpoint normalizers are finite 1x1x85x85 fields'))
    except Exception:
        checks.append((names[1], False, traceback.format_exc()))

    try:
        lifting = state['lifting.0.weight']
        channels, in_channels = map(int, lifting.shape[:2])
        layer_ids = sorted({int(k.split('.')[1]) for k in state if k.startswith('attention_layers.')})
        queries = state['attention_layers.0.slice_queries']
        heads, slices, head_dim = map(int, queries.shape)
        radon_ids = tuple(sorted({int(k.split('.')[1]) for k in state if k.startswith('radon_layers.')}))
        idx = radon_ids[0]
        derived = {
            'in_channels': in_channels, 'out_channels': int(state['projection.3.weight'].shape[0]),
            'channels': channels, 'layers': len(layer_ids), 'heads': heads,
            'slices': slices, 'angles': int(state[f'radon_layers.{idx}.radon.angles'].numel()),
            'radon_indices': radon_ids,
            'angle_hidden': int(state[f'radon_layers.{idx}.weights.network.0.weight'].shape[0]),
            'mlp_ratio': float(state['attention_layers.0.mlp.0.weight'].shape[0]) / channels,
        }
        if layer_ids != list(range(8)) or heads * head_dim != channels or derived != EXPECTED:
            raise AssertionError(f'RNO state-derived architecture differs: {derived}')
        saved = payload.get('config')
        saved_model = saved.get('model', {}) if isinstance(saved, Mapping) else {}
        if isinstance(saved_model, Mapping):
            for name, value in derived.items():
                if name in saved_model and (tuple(saved_model[name]) if name == 'radon_indices' else saved_model[name]) != value:
                    raise AssertionError(f'checkpoint saved config {name} disagrees with state shapes')
        checks.append((names[2], True, f'state-derived RNO parameters {derived}'))
    except Exception:
        checks.append((names[2], False, traceback.format_exc()))
        return checks + [(name, False, 'checkpoint architecture derivation failed') for name in names[3:]]

    try:
        module = _load_candidate_modules('model.py')
        factory = getattr(module, 'build_model_from_state_dict', None)
        saved = payload.get('config', {})
        saved_model = saved.get('model', {}) if isinstance(saved, Mapping) else {}
        if callable(factory):
            sig = inspect.signature(factory)
            positional = [p for p in sig.parameters.values()
                          if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            built = factory(state, saved_model) if len(positional) >= 2 else factory(state)
            model = built[0] if isinstance(built, tuple) else built
        else:
            cls = getattr(module, 'RadonNeuralOperator2D', None)
            if cls is None:
                raise AttributeError('expected RadonNeuralOperator2D or build_model_from_state_dict')
            sig = inspect.signature(cls)
            args = {**EXPECTED, 'hidden_dim': 128, 'num_layers': 8, 'num_heads': 8,
                    'num_slices': 32, 'num_angles': 32, 'radon_blocks': 4,
                    'angle_hidden_channels': 16, 'dropout': 0.0}
            if 'config' in sig.parameters:
                args['config'] = saved_model
            kwargs = {k: v for k, v in args.items() if k in sig.parameters}
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                kwargs = {k: v for k, v in EXPECTED.items()}
            model = cls(**kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError('RNO model must inherit torch.nn.Module')
        model.load_state_dict(state, strict=True)
        actual = model.state_dict()
        if set(actual) != set(state) or any(
            actual[key].shape != state[key].shape
            or actual[key].dtype != state[key].dtype
            or not torch.equal(actual[key].cpu(), state[key].cpu())
            for key in state
        ):
            raise AssertionError('strict-loaded model must match all 178 original tensors')
        checks.append((names[3], True, 'all 178 state tensors strict=True loaded with exact values'))
    except Exception:
        checks.append((names[3], False, traceback.format_exc()))

    try:
        # Verify candidate's own strict-loader behavior, not merely the test
        # harness calling strict=True on an independently constructed model.
        trees = [ast.parse((CANDIDATE_DIR / file).read_text(encoding='utf-8'))
                 for file in ('model.py', 'inference.py')]
        declared_strict = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'load_state_dict'
            and any(kw.arg == 'strict' and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True for kw in node.keywords)
            for tree in trees for node in ast.walk(tree)
        )
        factory = getattr(_load_candidate_modules('model.py'), 'build_model_from_state_dict', None)
        rejects_incomplete = False
        if callable(factory):
            incomplete = dict(state)
            incomplete.pop(next(iter(incomplete)))
            try:
                signature = inspect.signature(factory)
                required = [p for p in signature.parameters.values()
                            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
                            and p.default is inspect.Parameter.empty]
                if len(required) >= 2:
                    factory(incomplete, saved_model)
                else:
                    factory(incomplete)
            except (RuntimeError, KeyError, ValueError, AssertionError):
                rejects_incomplete = True
        if not (declared_strict or rejects_incomplete):
            raise AssertionError('candidate has neither explicit strict=True load nor a factory rejecting incomplete state')
        checks.append((names[4], True,
                       f'candidate explicit strict load={declared_strict}; rejects missing state={rejects_incomplete}'))
    except Exception:
        checks.append((names[4], False, traceback.format_exc()))
    return checks


def check_model_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Use common declarative engine plus exact checkpoint-derived layer topology."""
    try:
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        state = next(payload[k] for k in ('model', 'model_state_dict', 'state_dict')
                     if isinstance(payload.get(k), Mapping))
        mod = _load_candidate_modules('model.py')
        factory = getattr(mod, 'build_model_from_state_dict', None)
        saved = payload.get('config', {})
        saved_model = saved.get('model', {}) if isinstance(saved, Mapping) else {}
        if callable(factory):
            sig = inspect.signature(factory)
            positional = [p for p in sig.parameters.values()
                          if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            built = factory(state, saved_model) if len(positional) >= 2 else factory(state)
            model = built[0] if isinstance(built, tuple) else built
        else:
            cls = getattr(mod, 'RadonNeuralOperator2D', None)
            if cls is None:
                raise AttributeError('expected RadonNeuralOperator2D or build_model_from_state_dict')
            sig = inspect.signature(cls)
            args = {**EXPECTED, 'hidden_dim': 128, 'num_layers': 8, 'num_heads': 8,
                    'num_slices': 32, 'num_angles': 32, 'radon_blocks': 4,
                    'angle_hidden_channels': 16, 'dropout': 0.0}
            if 'config' in sig.parameters:
                args['config'] = saved_model
            kwargs = {k: v for k, v in args.items() if k in sig.parameters}
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                kwargs = {k: v for k, v in EXPECTED.items()}
            model = cls(**kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError('RNO model must inherit torch.nn.Module')
        model.load_state_dict(state, strict=True)
        checks = [('model_structure:checkpoint_model', True, 'checkpoint-derived model constructed and strictly loaded')]
    except Exception:
        return [('model_structure:checkpoint_model', False, traceback.format_exc())]

    try:
        rules = config.get('rules', [])
        if not rules:
            raise ValueError('model_structure.rules must not be empty')
        checks.extend(checkpoints_to_tuples(check_model_structure_v2(
            model, config_dict={'rules': rules}, verbose=False)))
    except Exception:
        checks.append(('model_structure:declared_rules', False, traceback.format_exc()))

    try:
        attention = getattr(model, 'attention_layers', None)
        radon = getattr(model, 'radon_layers', None)
        if attention is None or len(attention) != 8 or radon is None or len(radon) != 4:
            raise AssertionError('expected 8 attention layers and 4 Radon blocks')
        indices = tuple(sorted(int(k) for k in radon.keys()))
        if indices != (1, 3, 5, 7):
            raise AssertionError(f'Radon positions must be 1,3,5,7; got {indices}')
        registered = model.state_dict()
        if set(registered) != set(state) or any(registered[k].shape != state[k].shape for k in state):
            raise AssertionError('registered layer state differs from checkpoint')
        checks.append(('model_structure:radon_interleaving', True, '8 attention layers, 4 Radon blocks at 1/3/5/7'))
    except Exception:
        checks.append(('model_structure:radon_interleaving', False, traceback.format_exc()))
    return checks


def check_model_forward_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check output, repeatability, layout and batch independence on real loaded weights."""
    names = ('model_forward:output_shape', 'model_forward:finite',
             'model_forward:deterministic', 'model_forward:batch_independence')
    try:
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        state = next(payload[k] for k in ('model', 'model_state_dict', 'state_dict')
                     if isinstance(payload.get(k), Mapping))
        mod = _load_candidate_modules('model.py')
        factory = getattr(mod, 'build_model_from_state_dict', None)
        saved = payload.get('config', {})
        saved_model = saved.get('model', {}) if isinstance(saved, Mapping) else {}
        if callable(factory):
            sig = inspect.signature(factory)
            positional = [p for p in sig.parameters.values()
                          if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            built = factory(state, saved_model) if len(positional) >= 2 else factory(state)
            model = built[0] if isinstance(built, tuple) else built
        else:
            cls = getattr(mod, 'RadonNeuralOperator2D', None)
            if cls is None:
                raise AttributeError('expected RadonNeuralOperator2D or build_model_from_state_dict')
            sig = inspect.signature(cls)
            args = {**EXPECTED, 'hidden_dim': 128, 'num_layers': 8, 'num_heads': 8,
                    'num_slices': 32, 'num_angles': 32, 'radon_blocks': 4,
                    'angle_hidden_channels': 16, 'dropout': 0.0}
            if 'config' in sig.parameters:
                args['config'] = saved_model
            kwargs = {k: v for k, v in args.items() if k in sig.parameters}
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                kwargs = {k: v for k, v in EXPECTED.items()}
            model = cls(**kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError('RNO model must inherit torch.nn.Module')
        model.load_state_dict(state, strict=True)
        model.cpu().eval()
        torch.manual_seed(int(config.get('seed', 20260826)))
        batch = max(2, int(config.get('batch_sizes', [1])[-1]))
        height, width = map(int, config.get('spatial_shapes', [[85, 85]])[0])
        x = torch.randn(batch, 1, height, width)
        outputs = []
        with torch.inference_mode():
            for z in [x, x, *[x[j:j+1] for j in range(batch)]]:
                b, _, h, w = z.shape
                yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing='ij')
                coords = torch.stack((xx, yy), dim=-1).reshape(1, h*w, 2).expand(b, -1, -1)
                field = z.flatten(2).transpose(1, 2)
                failures = []
                for args in ((z,), (coords, field), (field, coords)):
                    try:
                        y = model(*args)
                        if isinstance(y, (tuple, list)):
                            y = y[0]
                        if isinstance(y, Mapping):
                            y = next(y[k] for k in ('pred', 'prediction', 'output', 'sol') if k in y)
                        if y.shape == (b, h, w, 1):
                            y = y.permute(0, 3, 1, 2)
                        elif y.shape == (b, h*w, 1):
                            y = y.transpose(1, 2).reshape(b, 1, h, w)
                        elif y.shape == (b, h, w):
                            y = y[:, None]
                        if y.shape != (b, 1, h, w) or not torch.isfinite(y).all():
                            raise ValueError(f'invalid forward output: {tuple(y.shape)}')
                        outputs.append(y)
                        break
                    except Exception as exc:
                        failures.append(str(exc))
                else:
                    raise RuntimeError('candidate forward conventions failed: '+str(failures))
        output, repeat = outputs[:2]
        separate = torch.cat(outputs[2:], dim=0)
        if x.shape != (batch, 1, height, width):
            raise AssertionError('wrong input')
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]
    checks = [(names[0], tuple(output.shape) == tuple(x.shape), f'B={batch}, H={height}, W={width}; output={tuple(output.shape)}'),
              (names[1], bool(torch.isfinite(output).all()), 'all output values finite')]
    try:
        torch.testing.assert_close(output, repeat, atol=1e-6, rtol=1e-5)
        checks.append((names[2], True, 'repeated inference numerically deterministic'))
    except Exception:
        checks.append((names[2], False, traceback.format_exc()))
    try:
        torch.testing.assert_close(output, separate, atol=1e-6, rtol=1e-5)
        checks.append((names[3], True, 'batch inference matches independent singleton predictions'))
    except Exception:
        checks.append((names[3], False, traceback.format_exc()))
    return checks


def check_model_forward_equivalence(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Run candidate/reference with exact fixed state on non-square probes."""
    names = ('model_forward_equivalence:reference_model',
             'model_forward_equivalence:multi_shape_numeric',
             'model_forward_equivalence:checkpoint_integrity')
    try:
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        state = next(payload[k] for k in ('model', 'model_state_dict', 'state_dict')
                     if isinstance(payload.get(k), Mapping))
        models = []
        for loader in (_load_reference_modules, _load_candidate_modules):
            mod = loader('model.py')
            factory = getattr(mod, 'build_model_from_state_dict', None)
            saved = payload.get('config', {})
            saved_model = saved.get('model', {}) if isinstance(saved, Mapping) else {}
            if callable(factory):
                sig = inspect.signature(factory)
                positional = [p for p in sig.parameters.values()
                              if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
                built = factory(state, saved_model) if len(positional) >= 2 else factory(state)
                model = built[0] if isinstance(built, tuple) else built
            else:
                cls = getattr(mod, 'RadonNeuralOperator2D', None)
                if cls is None:
                    raise AttributeError('expected RadonNeuralOperator2D or build_model_from_state_dict')
                sig = inspect.signature(cls)
                args = {**EXPECTED, 'hidden_dim': 128, 'num_layers': 8, 'num_heads': 8,
                        'num_slices': 32, 'num_angles': 32, 'radon_blocks': 4,
                        'angle_hidden_channels': 16, 'dropout': 0.0}
                if 'config' in sig.parameters:
                    args['config'] = saved_model
                kwargs = {k: v for k, v in args.items() if k in sig.parameters}
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    kwargs = {k: v for k, v in EXPECTED.items()}
                model = cls(**kwargs)
            if not isinstance(model, torch.nn.Module):
                raise TypeError('RNO model must inherit torch.nn.Module')
            model.load_state_dict(state, strict=True)
            models.append(model.cpu().eval())
        checks = [(names[0], True, 'candidate and independent standard loaded identical fixed checkpoint')]
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    try:
        torch.manual_seed(int(config.get('seed', 20260826)))
        maxima = []
        for batch, (height, width) in zip(config.get('batch_sizes', [1]), config.get('spatial_shapes', [[85, 85]])):
            batch, height, width = int(batch), int(height), int(width)
            x = torch.randn(batch, 1, height, width)
            fields = x.flatten(2).transpose(1, 2)
            yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij')
            coords = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2).expand(batch, -1, -1)
            values = []
            with torch.inference_mode():
                for model in models:
                    errors = []
                    for args in ((x,), (coords, fields), (fields, coords)):
                        try:
                            y = model(*args)
                            if isinstance(y, (list, tuple)):
                                y = y[0]
                            if isinstance(y, Mapping):
                                y = next(y[k] for k in ('pred', 'prediction', 'output', 'sol') if k in y)
                            if y.shape == (batch, height, width, 1):
                                y = y.permute(0, 3, 1, 2)
                            elif y.shape == (batch, height*width, 1):
                                y = y.transpose(1, 2).reshape(batch, 1, height, width)
                            elif y.shape == (batch, height, width):
                                y = y[:, None]
                            if y.shape != x.shape or not torch.isfinite(y).all():
                                raise ValueError('wrong output shape/nonfinite')
                            values.append(y)
                            break
                        except Exception as exc:
                            errors.append(str(exc))
                    else:
                        raise RuntimeError(f'forward conventions failed: {errors}')
            a, b = values
            torch.testing.assert_close(a, b, atol=float(config.get('atol', 1e-6)), rtol=float(config.get('rtol', 1e-5)))
            maxima.append(f'B={batch},HxW={height}x{width},max_abs={float((a-b).abs().max()):.9g}')
        checks.append((names[1], True, '; '.join(maxima)))
    except Exception:
        checks.append((names[1], False, traceback.format_exc()))

    try:
        for model in models:
            for key, tensor in state.items():
                torch.testing.assert_close(model.state_dict()[key].cpu(), tensor.cpu(), atol=0, rtol=0)
            if any(part.training for part in model.modules()):
                raise AssertionError('model not in eval mode')
        checks.append((names[2], True, 'both model state_dicts unchanged and in eval mode'))
    except Exception:
        checks.append((names[2], False, traceback.format_exc()))
    return checks


def check_infer_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Execute the submission's real inference entrypoint; no stale summary accepted."""
    name = 'infer_execution:fresh_first_batch_cli'
    try:
        output = (CANDIDATE_DIR / config.get('output_path', 'inference_summary.json')).resolve()
        if not output.is_relative_to(CANDIDATE_DIR.resolve()):
            raise ValueError('output must remain within candidate directory')
        script = CANDIDATE_DIR / 'inference.py'
        if output.exists():
            output.unlink()
        # This prompt does not mandate CLI flags: use only explicitly advertised flags.
        # --help itself might execute the program in poorly implemented CLIs;
        # delete any side-effect file again before the evaluated run.
        try:
            help_run = subprocess.run([sys.executable, str(script), '--help'], cwd=CANDIDATE_DIR,
                                      capture_output=True, text=True, timeout=30, check=False)
            documented = help_run.stdout + help_run.stderr if help_run.returncode == 0 else ''
        except subprocess.TimeoutExpired:
            documented = ''  # Help flags are optional, never a prerequisite of the prompt.
        if output.exists():
            output.unlink()
        args = [sys.executable, str(script)]
        for choices, value in (
            (('--checkpoint', '--ckpt'), str(CHECKPOINT)),
            (('--data', '--data-path', '--input', '--mat-path'), str(DATA)),
            (('--output', '--output-path'), str(output)),
            (('--device',), 'cuda:0' if torch.cuda.is_available() else 'cpu'),
            (('--batch-size', '--batch_size'), '8'),
            (('--samples', '--num-samples'), '200'),
            (('--stride', '--subsample-stride'), '5'),
        ):
            match = next((flag for flag in choices if re.search(r'(?<![\w-])' + re.escape(flag) + r'(?=[=\s\]\[,]|$)', documented)), None)
            if match is not None:
                args.extend((match, value))
        result = subprocess.run(args, cwd=CANDIDATE_DIR, capture_output=True, text=True,
                                timeout=int(config.get('timeout', 1200)), check=False)
        if result.returncode:
            raise RuntimeError(f'CLI returncode={result.returncode}\nstdout={result.stdout[-3000:]}\nstderr={result.stderr[-6000:]}')
        if not output.is_file() or not output.stat().st_size:
            raise AssertionError('CLI did not create fresh inference_summary.json')
        summary = json.loads(output.read_text(encoding='utf-8'))
        if not isinstance(summary, Mapping) or not summary:
            raise ValueError('summary must be a nonempty JSON mapping')
        return [(name, True, f'generated fresh summary via {args}')]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_infer_output_schema(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Reuse the generic artifact loader then inspect task-specific physical metrics."""
    checks = list(_base_check_infer_output_schema(config, caller_globals=caller_globals,
                                                   caller_locals=caller_locals))
    try:
        output = CANDIDATE_DIR / config.get('output_path', 'inference_summary.json')
        obj = json.loads(output.read_text(encoding='utf-8'))
        if not isinstance(obj, Mapping):
            raise TypeError('JSON summary must be an object')
        expected_count = int(config.get('expected_samples', 8))
        if 'selected_samples' in obj and obj['selected_samples'] != 200:
            raise AssertionError('data split must select the first 200 samples')
        if 'subsample_stride' in obj and obj['subsample_stride'] != 5:
            raise AssertionError('spatial downsampling stride must be 5')
        if 'downsampled_shape' in obj and obj['downsampled_shape'] != [200, 1, 85, 85]:
            raise AssertionError('selected data must be (200,1,85,85)')
        if 'shuffle' in obj and obj['shuffle'] is not False:
            raise AssertionError('test loader must not shuffle')
        if 'evaluated_samples' in obj and obj['evaluated_samples'] != expected_count:
            raise AssertionError('must evaluate first 8 samples')
        if 'batch_size' in obj and obj['batch_size'] != expected_count:
            raise AssertionError('first test batch must have batch_size 8')
        if 'evaluated_batch_index' in obj and obj['evaluated_batch_index'] != 0:
            raise AssertionError('only batch index 0 may be evaluated')
        if 'prediction_shape' in obj and tuple(obj['prediction_shape']) not in ((8, 1, 85, 85), (8, 85, 85, 1), (8, 85, 85)):
            raise AssertionError('wrong prediction spatial shape')
        if 'metrics_space' in obj and 'physical' not in str(obj['metrics_space']).lower():
            raise AssertionError('metrics must be in physical space')
        mse = next((obj[k] for k in ('mse_batch_mean', 'mse', 'MSE') if k in obj), None)
        rel = next((obj[k] for k in ('relative_l2_batch_mean', 'relative_l2', 'Relative_L2') if k in obj), None)
        if mse is None or rel is None:
            raise KeyError('missing MSE / Relative L2 batch mean')
        for name, value in (('mse', mse), ('relative_l2', rel)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise AssertionError(f'{name} batch mean must be a finite nonnegative scalar')
        checks.append(('infer_output_schema:physical_batch_metrics', True, f'MSE={mse}, Relative_L2={rel}'))
    except Exception:
        checks.append(('infer_output_schema:physical_batch_metrics', False, traceback.format_exc()))
    try:
        obj = json.loads((CANDIDATE_DIR / config.get('output_path', 'inference_summary.json')).read_text())
        for name, scalar_aliases, seq_aliases in (
            ('mse', ('mse_batch_mean', 'mse', 'MSE'), ('mse_per_sample',)),
            ('relative_l2', ('relative_l2_batch_mean', 'relative_l2', 'Relative_L2'), ('relative_l2_per_sample',))):
            scalars = [float(obj[k]) for k in scalar_aliases if k in obj]
            if not scalars or any(not math.isclose(v, scalars[0], rel_tol=1e-6, abs_tol=1e-9) for v in scalars):
                raise AssertionError(f'{name} missing or contradictory scalar aliases')
            values = next((obj[k] for k in seq_aliases if k in obj), None)
            if values is not None:
                if not isinstance(values, list) or len(values) != 8:
                    raise AssertionError(f'{name} per-sample metrics must have length 8')
                seq = [float(v) for v in values]
                if any(not math.isfinite(v) or v < 0 for v in seq) or not math.isclose(sum(seq)/8, scalars[0], rel_tol=1e-5, abs_tol=1e-7):
                    raise AssertionError(f'{name} per-sample mean mismatch')
        checks.append(('infer_output_schema:metric_self_consistency', True,
                       'metric scalar aliases/per-sample means agree whenever present'))
    except Exception:
        checks.append(('infer_output_schema:metric_self_consistency', False, traceback.format_exc()))
    return checks


def check_evaluate_oracle(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Independent first-batch, checkpoint-normalized physical-space metric oracle."""
    names = ('infer_oracle:first_200_stride5_protocol', 'infer_oracle:checkpoint_statistics',
             'infer_oracle:first_batch_forward_equivalence', 'infer_oracle:physical_mse',
             'infer_oracle:physical_relative_l2', 'infer_oracle:batch_metric_consistency')
    try:
        import scipy.io as sio
        try:
            obj = sio.loadmat(DATA, variable_names=('coeff', 'sol'))
            input_array, target_array = obj['coeff'], obj['sol']
        except (ValueError, NotImplementedError):
            import h5py
            with h5py.File(DATA, 'r') as mat:
                input_array, target_array = np.asarray(mat['coeff']), np.asarray(mat['sol'])
        if input_array.shape == (421, 421, 1024):
            input_array = input_array.transpose(2, 0, 1)
        if target_array.shape == (421, 421, 1024):
            target_array = target_array.transpose(2, 0, 1)
        if input_array.shape != (1024, 421, 421) or target_array.shape != input_array.shape:
            raise AssertionError(f'unexpected raw Darcy MAT shapes {input_array.shape}, {target_array.shape}')
        selected = int(config.get('first_samples', 200))
        stride = int(config.get('stride', 5))
        batch_size = int(config.get('batch_size', 8))
        if (selected, stride, batch_size) != (200, 5, 8):
            raise AssertionError('fixed data protocol must be 200 / 5 / 8')
        inputs = torch.from_numpy(np.ascontiguousarray(input_array[:batch_size, ::stride, ::stride], dtype=np.float32)).unsqueeze(1)
        targets = torch.from_numpy(np.ascontiguousarray(target_array[:batch_size, ::stride, ::stride], dtype=np.float32)).unsqueeze(1)
        if inputs.shape != (8, 1, 85, 85) or targets.shape != inputs.shape:
            raise AssertionError('first batch shape must be (8,1,85,85)')
        del obj, input_array, target_array
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise AssertionError('MAT first batch has nonfinite coefficient/solution')
        checks = [(names[0], True, 'first eight of first 200, coeff/sol, ::5, unshuffled')]
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    try:
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        state = next(payload[k] for k in ('model', 'model_state_dict', 'state_dict') if isinstance(payload.get(k), Mapping))
        states = []
        for key in ('input_normalizer', 'output_normalizer'):
            stats = payload[key]
            mean = torch.as_tensor(stats['mean']).float()
            std = torch.as_tensor(stats['std']).float()
            eps = float(stats.get('eps', 1e-6))
            if mean.shape != (1, 1, 85, 85) or std.shape != mean.shape or eps <= 0:
                raise AssertionError(f'incorrect checkpoint normalizer {key}')
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
                raise AssertionError(f'nonfinite checkpoint normalizer {key}')
            states.append((mean, std, eps))
        checks.append((names[1], True, 'both checkpoint statistics restored without refitting test data'))
    except Exception:
        return checks + [(name, False, traceback.format_exc()) for name in names[1:]]

    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        models = []
        for loader in (_load_reference_modules, _load_candidate_modules):
            mod = loader('model.py')
            factory = getattr(mod, 'build_model_from_state_dict', None)
            saved = payload.get('config', {})
            saved_model = saved.get('model', {}) if isinstance(saved, Mapping) else {}
            if callable(factory):
                sig = inspect.signature(factory)
                positional = [p for p in sig.parameters.values()
                              if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
                built = factory(state, saved_model) if len(positional) >= 2 else factory(state)
                model = built[0] if isinstance(built, tuple) else built
            else:
                cls = getattr(mod, 'RadonNeuralOperator2D', None)
                if cls is None:
                    raise AttributeError('expected RadonNeuralOperator2D or build_model_from_state_dict')
                sig = inspect.signature(cls)
                args = {**EXPECTED, 'hidden_dim': 128, 'num_layers': 8, 'num_heads': 8,
                        'num_slices': 32, 'num_angles': 32, 'radon_blocks': 4,
                        'angle_hidden_channels': 16, 'dropout': 0.0}
                if 'config' in sig.parameters:
                    args['config'] = saved_model
                kwargs = {k: v for k, v in args.items() if k in sig.parameters}
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    kwargs = {k: v for k, v in EXPECTED.items()}
                model = cls(**kwargs)
            if not isinstance(model, torch.nn.Module):
                raise TypeError('RNO model must inherit torch.nn.Module')
            model.load_state_dict(state, strict=True)
            models.append(model.to(device).eval())
        (input_mean, input_std, input_eps), (output_mean, output_std, output_eps) = states
        normalized = ((inputs - input_mean)/(input_std+input_eps)).to(device)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 85, device=device),
                                torch.linspace(0, 1, 85, device=device), indexing='ij')
        positions = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2).expand(8, -1, -1)
        field = normalized.flatten(2).transpose(1, 2)
        predictions = []
        with torch.inference_mode():
            for model in models:
                errors = []
                for args in ((normalized,), (positions, field), (field, positions)):
                    try:
                        y = model(*args)
                        if isinstance(y, (tuple, list)):
                            y = y[0]
                        if isinstance(y, Mapping):
                            y = next(y[k] for k in ('pred', 'prediction', 'output', 'sol') if k in y)
                        if y.shape == (8, 85, 85, 1):
                            y = y.permute(0, 3, 1, 2)
                        elif y.shape == (8, 85*85, 1):
                            y = y.transpose(1, 2).reshape(8, 1, 85, 85)
                        elif y.shape == (8, 85, 85):
                            y = y[:, None]
                        if y.shape != (8, 1, 85, 85) or not torch.isfinite(y).all():
                            raise AssertionError('model returned wrong shape or nonfinite data')
                        predictions.append((y.cpu() * (output_std + output_eps) + output_mean).double())
                        break
                    except Exception as exc:
                        errors.append(str(exc))
                else:
                    raise RuntimeError(f'all forward calling conventions failed: {errors}')
        reference, candidate = predictions
        torch.testing.assert_close(candidate, reference,
                                   atol=float(config.get('forward_atol', 1e-6)),
                                   rtol=float(config.get('forward_rtol', 1e-5)))
        checks.append((names[2], True, f'first eight physical predictions match independent reference; maxabs={float((candidate-reference).abs().max()):.9g}'))
    except Exception:
        return checks + [(name, False, traceback.format_exc()) for name in names[2:]]
    finally:
        if 'models' in locals():
            for model in models:
                model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        summary = json.loads((CANDIDATE_DIR / 'inference_summary.json').read_text(encoding='utf-8'))
        if not isinstance(summary, Mapping):
            raise TypeError('summary must be a JSON mapping')
        saved_mse = next(float(summary[k]) for k in ('mse_batch_mean','mse','MSE') if k in summary)
        saved_rel = next(float(summary[k]) for k in ('relative_l2_batch_mean','relative_l2','Relative_L2') if k in summary)
        target = targets.double()
        errors = candidate - target
        mse_per_sample = errors.square().flatten(1).mean(dim=1)
        relative_per_sample = (torch.linalg.vector_norm(errors.flatten(1), dim=1)
                               / torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-12))
        mse_true, rel_true = float(mse_per_sample.mean()), float(relative_per_sample.mean())
        if not math.isclose(saved_mse, mse_true, rel_tol=float(config.get('rtol', 1e-4)), abs_tol=float(config.get('atol', 1e-6))):
            raise AssertionError(f'MSE summary={saved_mse}, independently calculated={mse_true}')
        checks.append((names[3], True, f'physical grid point MSE={mse_true:.9g}'))
    except Exception:
        checks.append((names[3], False, traceback.format_exc()))
    try:
        if not math.isclose(saved_rel, rel_true, rel_tol=float(config.get('rtol', 1e-4)), abs_tol=float(config.get('atol', 1e-6))):
            raise AssertionError(f'Relative L2 summary={saved_rel}, independently calculated={rel_true}')
        checks.append((names[4], True, f'per-sample flattened Relative L2 mean={rel_true:.9g}'))
    except Exception:
        checks.append((names[4], False, traceback.format_exc()))
    try:
        for name, computed in (('mse_per_sample', mse_per_sample), ('relative_l2_per_sample', relative_per_sample)):
            if name not in summary:
                continue  # prompt requires batch means, not per-sample fields.
            observed = torch.as_tensor(summary[name], dtype=torch.float64)
            torch.testing.assert_close(observed, computed, rtol=float(config.get('rtol', 1e-4)),
                                       atol=float(config.get('atol', 1e-6)))
        if 'metrics_space' in summary and 'physical' not in str(summary['metrics_space']).lower():
            raise AssertionError('summary reports normalized-space metrics')
        checks.append((names[5], True, 'saved optional per-sample metrics and physical-space claims consistent'))
    except Exception:
        checks.append((names[5], False, traceback.format_exc()))
    return checks
