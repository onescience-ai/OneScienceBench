#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepCFD inference evaluator: shared checks plus task-specific check methods.

The runner configures candidate/reference directories before importing the
shared evaluator. There is deliberately no environment/bootstrap logic here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import pickle
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml

from base.base_evaluator import (
    CANDIDATE_DIR,
    _load_candidate_modules,
    _load_reference_modules,
    _load_class,
    check_files,
    check_interface,
    check_config as _base_check_config,
    check_infer_structure as _base_check_infer_structure,
    check_infer_output_schema as _base_check_infer_output_schema,
)
from base.model_structure_checker import check_model_structure_v2, checkpoints_to_tuples

DATA_ROOT = Path('/public/share/sugonhpcapp01/onestore/onedatasets/DeepCFD')
CHECKPOINT_PATH = DATA_ROOT / 'weight/best_model.pt'
DATA_DIR = DATA_ROOT / 'data'
METRIC_KEYS = ('Ux_MSE', 'Uy_MSE', 'p_MSE', 'Total_MSE')
SHAPE = (3, 172, 79)
RTOL, ATOL = 1e-6, 1e-7
MODEL_CONFIG = {
    'in_channels': 3, 'out_channels': 3, 'base_channels': 8,
    'num_stages': 4, 'kernel_size': 5, 'bilinear': True, 'normtype': 'none',
}


def check_config_content(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check valid YAML and real data/weight/output semantics, not YAML layout."""
    checks = list(_base_check_config({'config_name': config.get('config_name', 'config.yaml')},
                                     caller_globals=caller_globals, caller_locals=caller_locals))
    try:
        content = yaml.safe_load((CANDIDATE_DIR / 'config.yaml').read_text(encoding='utf-8'))
        if not isinstance(content, Mapping):
            raise TypeError('config.yaml must be a YAML mapping')
        checks.append(('config_content:yaml_mapping', True, 'valid mapping'))
    except Exception:
        checks.append(('config_content:yaml_mapping', False, traceback.format_exc()))
        return checks

    # The prompt does not fix data/inference/model/output top-level key names.
    # If a resource is explicitly configured, verify its identity; otherwise
    # leave actual resource usage to the CLI and complete-test oracle.
    for name, trusted in (
        ('dataX.pkl', DATA_DIR / 'dataX.pkl'),
        ('dataY.pkl', DATA_DIR / 'dataY.pkl'),
        ('checkpoint', CHECKPOINT_PATH),
    ):
        try:
            sections = [content, *(value for value in content.values() if isinstance(value, Mapping))]
            if name == 'checkpoint':
                aliases = ('checkpoint_path', 'checkpoint', 'weight_path', 'weights_path')
                candidates = [section[key] for section in sections for key in aliases
                              if isinstance(section.get(key), (str, Path))]
                if not candidates:
                    checks.append(('config_content:checkpoint_identity', True,
                                   'not explicitly configured in YAML; validated at runtime'))
                    continue
                source_paths = [Path(str(value)).expanduser() for value in candidates]
            else:
                section = content.get('data', {})
                if not isinstance(section, Mapping):
                    raise TypeError('data section must be mapping when present')
                raw_dir = next((section[key] for key in ('directory', 'data_dir', 'dataset_dir')
                                if isinstance(section.get(key), (str, Path))), None)
                if raw_dir is None:
                    checks.append((f'config_content:{name}', True,
                                   'dataset path not explicitly configured; checked by execution/oracle'))
                    continue
                directory = Path(str(raw_dir)).expanduser()
                source_paths = [directory / name]
                if directory.name != 'data':
                    source_paths.append(directory / 'data' / name)
            matched = False
            for given in source_paths:
                given = given if given.is_absolute() else CANDIDATE_DIR / given
                given = given.resolve()
                if not given.is_file():
                    continue
                if not trusted.is_file():
                    raise FileNotFoundError(f'missing canonical DeepCFD resource: {trusted}')
                if given.samefile(trusted):
                    matched = True
                    break
                if given.stat().st_size != trusted.stat().st_size:
                    continue
                # Hash only same-size candidates; accepts a verified content copy.
                digests = []
                for path in (given, trusted):
                    sha = hashlib.sha256()
                    with path.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                            sha.update(chunk)
                    digests.append(sha.digest())
                if digests[0] == digests[1]:
                    matched = True
                    break
            if not matched:
                raise AssertionError(f'configured {name} is unavailable or differs from {trusted}; '
                                     f'paths={source_paths}')
            checks.append((f'config_content:{name}', True, 'matches canonical DeepCFD resource'))
        except Exception:
            checks.append((f'config_content:{name}', False, traceback.format_exc()))

    try:
        sections = [content, *(value for value in content.values() if isinstance(value, Mapping))]
        outputs = [section[key] for section in sections
                   for key in ('metrics_path', 'result_dir', 'output_dir')
                   if isinstance(section.get(key), (str, Path))]
        expected_dir = (CANDIDATE_DIR / 'result/deepcfd').resolve()
        expected_file = expected_dir / 'metrics.json'
        for output in outputs:
            chosen = Path(str(output)).expanduser()
            chosen = (chosen if chosen.is_absolute() else CANDIDATE_DIR / chosen).resolve()
            if chosen not in (expected_dir, expected_file):
                raise AssertionError(f'configured output is {chosen}, expected {expected_dir} or {expected_file}')
        checks.append(('config_content:output_routing', True,
                       'output path conforms or is validated by the CLI'))
    except Exception:
        checks.append(('config_content:output_routing', False, traceback.format_exc()))
    return checks


def check_model_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check meaningful U-Net topology using the shared structure rule engine."""
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        kwargs = dict(checkpoint['config'])
        kwargs.pop('name', None)
        cls = _load_class(_load_candidate_modules('unetex.py'), 'UNetEx')
        model = cls(**kwargs).cpu().eval()
        if not isinstance(model, torch.nn.Module):
            raise TypeError('UNetEx must inherit torch.nn.Module')
        checks = [('model_structure:build_model', True, 'constructed from fixed checkpoint config')]
    except Exception:
        return [('model_structure:build_model', False, traceback.format_exc())]

    try:
        rules = config.get('rules', [])
        if not rules:
            raise ValueError('missing structure rules')
        results = checkpoints_to_tuples(check_model_structure_v2(
            model, config_dict={'rules': rules}, verbose=False))
        checks.extend(results)
    except Exception:
        checks.append(('model_structure:rules', False, traceback.format_exc()))

    try:
        encoder = getattr(model, 'encoder', None)
        decoders = getattr(model, 'decoders', None)
        if not isinstance(encoder, torch.nn.Module):
            raise AssertionError('missing shared encoder')
        if not isinstance(decoders, torch.nn.Module) or len(list(decoders.children())) != 3:
            raise AssertionError('expected three Ux/Uy/p decoder branches')
        sets = [{id(param) for param in branch.parameters()} for branch in decoders.children()]
        if not all(sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError('decoder branches must have independent trainable parameters')
        if any({id(p) for p in encoder.parameters()} & branch for branch in sets):
            raise AssertionError('decoder parameters unexpectedly aliased to encoder')
        checks.append(('model_structure:semantic_roles', True,
                       'one encoder, three independent decoders'))
    except Exception:
        checks.append(('model_structure:semantic_roles', False, traceback.format_exc()))
    return checks


def check_checkpoint_loading(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check fixed-weight metadata and strict compatibility with both implementations."""
    names = ('checkpoint_loading:metadata', 'checkpoint_loading:reference_strict',
             'checkpoint_loading:candidate_state_layout', 'checkpoint_loading:candidate_strict')
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError('checkpoint must be a mapping')
        kwargs = dict(checkpoint.get('config') or {})
        kwargs.pop('name', None)
        state = next((checkpoint[key] for key in ('model_state', 'state_dict', 'model_state_dict')
                      if isinstance(checkpoint.get(key), Mapping)), None)
        if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError('checkpoint has no tensor state_dict')
        if kwargs != MODEL_CONFIG:
            raise AssertionError(f'checkpoint config={kwargs}, expected fixed DeepCFD {MODEL_CONFIG}')
        checks = [(names[0], True, f'checkpoint metadata valid, {len(state)} tensor keys')]
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    for label, filename in (('reference', names[1]), ('candidate', names[3])):
        try:
            loader = _load_reference_modules if label == 'reference' else _load_candidate_modules
            cls = _load_class(loader('unetex.py'), 'UNetEx')
            model = cls(**kwargs).cpu().eval()
            if not isinstance(model, torch.nn.Module):
                raise TypeError(f'{label} model is not torch.nn.Module')
            layout = model.state_dict()
            if label == 'candidate':
                if set(layout) != set(state):
                    raise AssertionError('candidate parameter keys do not match checkpoint')
                for key in state:
                    if layout[key].shape != state[key].shape or layout[key].dtype != state[key].dtype:
                        raise AssertionError(f'checkpoint tensor shape/dtype mismatch: {key}')
                checks.append((names[2], True, 'key/shape/dtype layout identical to checkpoint'))
            result = model.load_state_dict(state, strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise AssertionError(str(result))
            checks.append((filename, True, f'{label} strict=True load succeeds'))
        except Exception:
            if label == 'candidate' and not any(name == names[2] for name, *_ in checks):
                checks.append((names[2], False, traceback.format_exc()))
            checks.append((filename, False, traceback.format_exc()))
    return checks


def check_method_correctness(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Independent mathematical oracle for the prompt-mandated metric function."""
    if config.get('scope') != 'sample_normalized_grid_mse':
        return [('method_correctness:scope', False, 'wrong scope')]
    try:
        function = getattr(_load_candidate_modules('inference.py'),
                           'compute_sample_normalized_grid_mse')
        if not callable(function):
            raise TypeError('metric must be callable')
    except Exception:
        return [('method_correctness:metric_import', False, traceback.format_exc())]
    checks = []
    for batch, name in ((2, 'method_correctness:exact_grid_reduction'),
                        (5, 'method_correctness:batch_normalization')):
        try:
            rng = torch.Generator().manual_seed(20261010 + batch)
            predictions = torch.randn(batch, 3, 7, 5, generator=rng)
            targets = torch.randn(batch, 3, 7, 5, generator=rng)
            exact = ((predictions.double() - targets.double()).square()
                     .sum(dim=(0, 2, 3)) / batch).tolist()
            expected = dict(zip(METRIC_KEYS[:3], exact))
            expected['Total_MSE'] = sum(exact)
            actual = function(predictions, targets)
            if not isinstance(actual, Mapping) or set(actual) != set(METRIC_KEYS):
                raise AssertionError('metric must return exactly four named scalar values')
            for key, value in expected.items():
                if not math.isclose(float(actual[key]), value, rel_tol=RTOL, abs_tol=ATOL):
                    raise AssertionError(f'{key}: candidate={actual[key]}, expected={value}')
            checks.append((name, True, f'B={batch}, channel SSE / B matches independent oracle'))
        except Exception:
            checks.append((name, False, traceback.format_exc()))
    return checks


def check_model_forward_equivalence(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check loaded model forward values and inference-state stability."""
    names = ('model_forward:output_contract', 'model_forward:reference_equivalence',
             'model_forward:eval_state_integrity')
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        kwargs = dict(checkpoint['config'])
        kwargs.pop('name', None)
        state = next(checkpoint[key] for key in ('model_state', 'state_dict', 'model_state_dict')
                     if isinstance(checkpoint.get(key), Mapping))
        candidate = _load_class(_load_candidate_modules('unetex.py'), 'UNetEx')(**kwargs).cpu().eval()
        reference = _load_class(_load_reference_modules('unetex.py'), 'UNetEx')(**kwargs).cpu().eval()
        candidate.load_state_dict(state, strict=True)
        reference.load_state_dict(state, strict=True)
        batches = [int(value) for value in config.get('batch_sizes', [1, 2])]
        if not batches or any(value < 1 for value in batches):
            raise ValueError('batch sizes must be positive')
        rng = torch.Generator().manual_seed(int(config.get('seed', 42)))
        outputs = []
        with torch.inference_mode():
            for batch in batches:
                inputs = torch.randn(batch, *SHAPE, generator=rng)
                outputs.append((batch, candidate(inputs), reference(inputs)))
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    checks = []
    try:
        for batch, actual, _ in outputs:
            if not isinstance(actual, torch.Tensor) or tuple(actual.shape) != (batch, *SHAPE):
                raise AssertionError(f'wrong output shape at B={batch}')
            if not torch.isfinite(actual).all():
                raise AssertionError(f'nonfinite output at B={batch}')
        checks.append((names[0], True, 'BCHW shape and finite values verified'))
    except Exception:
        checks.append((names[0], False, traceback.format_exc()))
    try:
        maximum = 0.0
        for _, actual, expected in outputs:
            torch.testing.assert_close(actual, expected,
                                       atol=float(config.get('atol', 1e-6)),
                                       rtol=float(config.get('rtol', 1e-6)))
            maximum = max(maximum, float((actual.double() - expected.double()).abs().max()))
        checks.append((names[1], True, f'candidate/reference max_abs_diff={maximum:.9g}'))
    except Exception:
        checks.append((names[1], False, traceback.format_exc()))
    try:
        if any(layer.training for layer in candidate.modules()):
            raise AssertionError('eval mode not respected')
        before = {key: value.detach().clone() for key, value in candidate.state_dict().items()}
        values = torch.randn(1, *SHAPE, generator=rng)
        with torch.inference_mode():
            left, right = candidate(values), candidate(values)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        for key, value in before.items():
            torch.testing.assert_close(candidate.state_dict()[key], value, atol=0, rtol=0)
        checks.append((names[2], True, 'repeatability and weight integrity verified'))
    except Exception:
        checks.append((names[2], False, traceback.format_exc()))
    return checks


def check_infer_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Shared AST checks plus the explicitly requested metric invocation."""
    checks = list(_base_check_infer_structure(config, caller_globals=caller_globals,
                                              caller_locals=caller_locals))
    try:
        source = (CANDIDATE_DIR / 'inference.py').read_text(encoding='utf-8')
        nodes = list(ast.walk(ast.parse(source)))
        functions = {node.name for node in nodes
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        calls = {node.func.id for node in nodes
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        attrs = {node.func.attr for node in nodes
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        name = 'compute_sample_normalized_grid_mse'
        if name not in functions or name not in (calls | attrs):
            raise AssertionError('inference.py must define and invoke specified metric function')
        if '--config' not in source or 'eval' not in attrs:
            raise AssertionError('missing CLI --config flag or eval()')
        if not ({'no_grad', 'inference_mode', 'set_grad_enabled'} & attrs):
            raise AssertionError('inference must disable gradient recording')
        checks.append(('infer_structure:deepcfd_contract', True,
                       'required metric called and CLI/inference-mode semantics present'))
    except Exception:
        checks.append(('infer_structure:deepcfd_contract', False, traceback.format_exc()))
    return checks


def check_infer_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Run the exact user CLI; reject stale/invalid output."""
    try:
        script = CANDIDATE_DIR / 'inference.py'
        config_path = CANDIDATE_DIR / 'config.yaml'
        started = time.time_ns()
        result = subprocess.run(
            [sys.executable, str(script), '--config', str(config_path)],
            cwd=CANDIDATE_DIR, capture_output=True, text=True,
            timeout=int(config.get('timeout', 900)), check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f'CLI code={result.returncode}; stderr={result.stderr[-4000:]}')
        output = CANDIDATE_DIR / 'result/deepcfd/metrics.json'
        if not output.is_file() or output.stat().st_mtime_ns < started - 2_000_000_000:
            raise AssertionError('CLI did not generate fresh result/deepcfd/metrics.json')
        content = json.loads(output.read_text(encoding='utf-8'))
        if not isinstance(content, Mapping) or set(content) != set(METRIC_KEYS):
            raise ValueError('invalid metric output keys')
        if any(isinstance(content[key], bool) or not isinstance(content[key], (int, float)) or
               not math.isfinite(content[key]) or content[key] < 0 for key in METRIC_KEYS):
            raise ValueError('metrics must be finite nonnegative JSON numbers')
        if not math.isclose(content['Total_MSE'], sum(content[key] for key in METRIC_KEYS[:3]),
                            rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError('Total_MSE != sum of channel errors')
        return [('infer_execution:real_cli', True, 'CLI succeeded and produced fresh metrics')]
    except Exception:
        return [('infer_execution:real_cli', False, traceback.format_exc())]


def check_infer_output_schema(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    checks = list(_base_check_infer_output_schema(config, caller_globals=caller_globals,
                                                  caller_locals=caller_locals))
    try:
        data = json.loads((CANDIDATE_DIR / 'result/deepcfd/metrics.json').read_text(encoding='utf-8'))
        if not isinstance(data, Mapping) or set(data) != set(METRIC_KEYS):
            raise AssertionError('metrics.json must contain exactly four named metrics')
        if any(isinstance(data[key], bool) or not isinstance(data[key], (int, float)) or
               not math.isfinite(data[key]) or data[key] < 0 for key in METRIC_KEYS):
            raise AssertionError('invalid metric scalar value')
        if not math.isclose(data['Total_MSE'], sum(data[key] for key in METRIC_KEYS[:3]),
                            rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError('Total_MSE must be the three channel errors added')
        checks.append(('infer_output_schema:channel_metrics', True, 'all four metrics valid'))
    except Exception:
        checks.append(('infer_output_schema:channel_metrics', False, traceback.format_exc()))
    return checks


def check_evaluate_oracle(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Full DeepCFD benchmark split/forward/grid-SSE oracle, not YAML key matching."""
    names = ('infer_oracle:dataset_protocol', 'infer_oracle:full_test_forward',
             'infer_oracle:per_channel_grid_sse', 'infer_oracle:total_metric')
    try:
        with (DATA_DIR / 'dataX.pkl').open('rb') as stream:
            inputs = np.asarray(pickle.load(stream), dtype=np.float32)
        with (DATA_DIR / 'dataY.pkl').open('rb') as stream:
            targets = np.asarray(pickle.load(stream), dtype=np.float32)
        if inputs.shape != (981, *SHAPE) or targets.shape != (981, *SHAPE):
            raise AssertionError(f'canonical data shape mismatch {inputs.shape}, {targets.shape}')
        order = list(range(981))
        split_seed = int(config.get('seed', 0))
        random.Random(split_seed).shuffle(order)
        cut = int(981 * float(config.get('split_ratio', 0.7)))
        selected = order[cut:]
        if cut != 686 or len(selected) != 295:
            raise ValueError(f'benchmark split must be 686/295; got {cut}/{len(selected)}')
        protocol = f'981 samples, seed={split_seed}, train={cut}, test={len(selected)}'
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        kwargs = dict(checkpoint['config'])
        kwargs.pop('name', None)
        state = next(checkpoint[key] for key in ('model_state', 'state_dict', 'model_state_dict')
                     if isinstance(checkpoint.get(key), Mapping))
        candidate = _load_class(_load_candidate_modules('unetex.py'), 'UNetEx')(**kwargs).eval()
        reference = _load_class(_load_reference_modules('unetex.py'), 'UNetEx')(**kwargs).eval()
        candidate.load_state_dict(state, strict=True)
        reference.load_state_dict(state, strict=True)
        saved = json.loads((CANDIDATE_DIR / 'result/deepcfd/metrics.json').read_text(encoding='utf-8'))
        if not isinstance(saved, Mapping) or set(saved) != set(METRIC_KEYS):
            raise ValueError('missing four saved metrics')
    except Exception:
        return [(names[0], True, protocol)] + [
            (name, False, traceback.format_exc()) for name in names[1:]]

    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        candidate = candidate.to(device).eval()
        reference = reference.to(device).eval()
        channel_sum_candidate = torch.zeros(3, dtype=torch.float64)
        channel_sum_reference = torch.zeros(3, dtype=torch.float64)
        maximum = 0.0
        same = True
        batch_size = int(config.get('batch_size', 16))
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        with torch.inference_mode():
            for first in range(0, len(selected), batch_size):
                indices = selected[first:first + batch_size]
                xb = torch.from_numpy(inputs[indices].copy()).to(device)
                yb = torch.from_numpy(targets[indices].copy()).to(device)
                output = candidate(xb)
                expected = reference(xb)
                if output.shape != yb.shape or expected.shape != yb.shape:
                    raise AssertionError('full-test output tensor shape mismatch')
                if not (torch.isfinite(output).all() and torch.isfinite(expected).all()):
                    raise AssertionError('full-test nonfinite model output')
                maximum = max(maximum, float((output - expected).abs().max()))
                same &= bool(torch.allclose(output, expected,
                                            rtol=float(config.get('forward_rtol', RTOL)),
                                            atol=float(config.get('forward_atol', 1e-6))))
                channel_sum_candidate += ((output.double() - yb.double()).square()
                                          .sum(dim=(0, 2, 3)).cpu())
                channel_sum_reference += ((expected.double() - yb.double()).square()
                                          .sum(dim=(0, 2, 3)).cpu())
        channel_candidate = (channel_sum_candidate / len(selected)).tolist()
        channel_reference = (channel_sum_reference / len(selected)).tolist()
    except Exception:
        return [(names[0], True, protocol)] + [
            (name, False, traceback.format_exc()) for name in names[1:]]
    finally:
        candidate.cpu()
        reference.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checks = [(names[0], True, protocol),
              (names[1], same, f'full test max_abs_diff={maximum:.9g}')]
    try:
        for channel, key in enumerate(METRIC_KEYS[:3]):
            for label, value in (('candidate', channel_candidate[channel]),
                                 ('reference', channel_reference[channel])):
                if not math.isclose(float(saved[key]), value,
                                    rel_tol=float(config.get('rtol', RTOL)),
                                    abs_tol=float(config.get('atol', ATOL))):
                    raise AssertionError(f'{key}: saved={saved[key]}, {label}={value}')
        checks.append((names[2], True, 'all channel grid SSE / 295 match'))
    except Exception:
        checks.append((names[2], False, traceback.format_exc()))
    try:
        for label, vector in (('candidate', channel_candidate), ('reference', channel_reference)):
            expected_total = sum(vector)
            if not math.isclose(float(saved['Total_MSE']), expected_total,
                                rel_tol=float(config.get('rtol', RTOL)),
                                abs_tol=float(config.get('atol', ATOL))):
                raise AssertionError(f'Total_MSE saved={saved["Total_MSE"]}, '
                                     f'{label}={expected_total}')
        checks.append((names[3], True, 'Total_MSE matches complete-test reference'))
    except Exception:
        checks.append((names[3], False, traceback.format_exc()))
    return checks
