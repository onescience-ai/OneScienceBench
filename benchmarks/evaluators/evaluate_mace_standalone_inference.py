'''V2 adapter for the standalone MACE inference evaluator.'''
import ast
import builtins
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
from ase import Atoms
from ase.io import write

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


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['submission_artifact_discovery']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidates = []
        explicit = os.environ.get('MACE_STANDALONE_SUBMISSION_DIR')
        if explicit:
            candidates.append(Path(explicit).expanduser())
        SUBMISSION_DIR_HINTS = ('mace_standalone_inference', 'mace_standalone_inference_V1')
        SUBMISSION_PATH = (CANDIDATE_DIR / 'mace_standalone_inference.py').resolve()
        update_check(current_index, True, f'candidate module resolved to {SUBMISSION_PATH}')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_file(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['portable_model_file_exists', 'portable_model_sha256']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        MODEL_PATH = os.environ.get(
            'MACE_STANDALONE_MODEL_PATH',
            '/public/share/sugonhpcapp01/onestore/onemodels/mace/mace-mp-0-medium-portable.pt',
        )
        EXPECTED_MODEL_SHA256 = 'c53bab687dc91abf1084219ebf3ed55029f60708a53b7807752932d38c3680be'
        model_path = Path(MODEL_PATH).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f'portable model is unavailable: {model_path}')
        update_check(current_index, True, f'portable model is available: {model_path}')
        current_index += 1
        digest = hashlib.sha256()
        with model_path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != EXPECTED_MODEL_SHA256:
            raise ValueError(f'portable model SHA256 mismatch: {actual_hash} != {EXPECTED_MODEL_SHA256}')
        update_check(current_index, True, f'portable model SHA256 matches: {actual_hash}')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'standard_static_dependency_isolation',
        'standard_runtime_dependency_isolation',
        'candidate_static_dependency_isolation',
        'candidate_runtime_dependency_isolation',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        FORBIDDEN_ROOTS = {'onescience', 'mace'}
        original_import = builtins.__import__
        original_import_module = importlib.import_module

        def guarded_import(name: str, globals: dict[str, Any] | None = None,
                           locals: dict[str, Any] | None = None,
                           fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
            if name.split('.', 1)[0] in FORBIDDEN_ROOTS:
                raise ImportError(f'forbidden runtime import attempted: {name}')
            return original_import(name, globals, locals, fromlist, level)

        def guarded_import_module(name: str, package: str | None = None) -> Any:
            if name.split('.', 1)[0] in FORBIDDEN_ROOTS:
                raise ImportError(f'forbidden runtime import attempted: {name}')
            return original_import_module(name, package)

        builtins.__import__ = guarded_import
        importlib.import_module = guarded_import_module

        class _ForbiddenImportVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.errors: list[str] = []

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if alias.name.split('.', 1)[0] in FORBIDDEN_ROOTS:
                        self.errors.append(f'forbidden import: {alias.name}')
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module and node.module.split('.', 1)[0] in FORBIDDEN_ROOTS:
                    self.errors.append(f'forbidden import: {node.module}')
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                    self.errors.append('dynamic __import__ is forbidden')
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module':
                    self.errors.append('dynamic import_module is forbidden')
                self.generic_visit(node)

            def visit_Attribute(self, node: ast.Attribute) -> None:
                if node.attr == 'modules' and isinstance(node.value, ast.Name) and node.value.id == 'sys':
                    self.errors.append('sys.modules aliases are forbidden')
                self.generic_visit(node)

        standard_path = (REFERENCE_DIR / 'mace_standalone_inference.py').resolve()
        tree = ast.parse(standard_path.read_text(encoding='utf-8'), filename=str(standard_path))
        visitor = _ForbiddenImportVisitor()
        visitor.visit(tree)
        assert not visitor.errors, '; '.join(visitor.errors)
        update_check(current_index, True, f'standard static imports are isolated: {standard_path}')
        current_index += 1
        spec = importlib.util.spec_from_file_location('mace_standalone_inference_standard', standard_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load module: {standard_path}')
        standard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standard)
        update_check(current_index, True, 'standard module passed runtime import guard')
        current_index += 1

        if os.environ.get('MACE_STANDALONE_STANDARD_ONLY') == '1':
            candidate = standard
            update_check(current_index, True, 'candidate static validation uses the standard module')
            current_index += 1
            update_check(current_index, True, 'candidate runtime validation uses the standard module')
            current_index += 1
        else:
            SUBMISSION_PATH = (CANDIDATE_DIR / 'mace_standalone_inference.py').resolve()
            tree = ast.parse(SUBMISSION_PATH.read_text(encoding='utf-8'), filename=str(SUBMISSION_PATH))
            visitor = _ForbiddenImportVisitor()
            visitor.visit(tree)
            assert not visitor.errors, '; '.join(visitor.errors)
            update_check(current_index, True, f'candidate static imports are isolated: {SUBMISSION_PATH}')
            current_index += 1
            spec = importlib.util.spec_from_file_location('mace_standalone_inference_candidate', SUBMISSION_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f'cannot load module: {SUBMISSION_PATH}')
            candidate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate)
            update_check(current_index, True, 'candidate module passed runtime import guard')
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'standard_runner_class', 'standard_constructor_parameters', 'standard_run_method',
        'standard_data_format_default', 'standard_device_default',
        'candidate_runner_class', 'candidate_constructor_parameters', 'candidate_run_method',
        'candidate_data_format_default', 'candidate_device_default',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        standard_path = (REFERENCE_DIR / 'mace_standalone_inference.py').resolve()
        spec = importlib.util.spec_from_file_location('mace_standalone_inference_standard', standard_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load module: {standard_path}')
        standard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standard)
        runner_cls = getattr(standard, 'MACEStandaloneInferenceRunner', None)
        assert runner_cls is not None and callable(runner_cls), 'MACEStandaloneInferenceRunner is missing'
        update_check(current_index, True, 'standard MACEStandaloneInferenceRunner is present')
        current_index += 1
        signature = inspect.signature(runner_cls.__init__)
        required = {'model_path', 'input_path', 'output_dir', 'data_format', 'device'}
        missing = required - set(signature.parameters)
        assert not missing, f'runner constructor missing parameters: {sorted(missing)}'
        update_check(current_index, True, 'standard runner constructor parameters are present')
        current_index += 1
        assert hasattr(runner_cls, 'run') and callable(runner_cls.run), 'runner.run() is missing'
        update_check(current_index, True, 'standard runner.run() is callable')
        current_index += 1
        assert signature.parameters['data_format'].default == 'ase'
        update_check(current_index, True, 'standard data_format default is ase')
        current_index += 1
        assert signature.parameters['device'].default == 'cpu'
        update_check(current_index, True, 'standard device default is cpu')
        current_index += 1

        if os.environ.get('MACE_STANDALONE_STANDARD_ONLY') == '1':
            candidate = standard
        else:
            SUBMISSION_PATH = (CANDIDATE_DIR / 'mace_standalone_inference.py').resolve()
            spec = importlib.util.spec_from_file_location('mace_standalone_inference_candidate', SUBMISSION_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f'cannot load module: {SUBMISSION_PATH}')
            candidate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate)
        runner_cls = getattr(candidate, 'MACEStandaloneInferenceRunner', None)
        assert runner_cls is not None and callable(runner_cls), 'MACEStandaloneInferenceRunner is missing'
        update_check(current_index, True, 'candidate MACEStandaloneInferenceRunner is present')
        current_index += 1
        signature = inspect.signature(runner_cls.__init__)
        required = {'model_path', 'input_path', 'output_dir', 'data_format', 'device'}
        missing = required - set(signature.parameters)
        assert not missing, f'runner constructor missing parameters: {sorted(missing)}'
        update_check(current_index, True, 'candidate runner constructor parameters are present')
        current_index += 1
        assert hasattr(runner_cls, 'run') and callable(runner_cls.run), 'runner.run() is missing'
        update_check(current_index, True, 'candidate runner.run() is callable')
        current_index += 1
        assert signature.parameters['data_format'].default == 'ase'
        update_check(current_index, True, 'candidate data_format default is ase')
        current_index += 1
        assert signature.parameters['device'].default == 'cpu'
        update_check(current_index, True, 'candidate device default is cpu')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_execution(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'standard_ase_execution', 'ase_file_execution', 'ase_single_atoms_execution',
        'ase_atoms_sequence_execution', 'hdf5_file_execution', 'hdf5_sharded_dir_execution',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        MODEL_PATH = os.environ.get('MACE_STANDALONE_MODEL_PATH', '/public/share/sugonhpcapp01/onestore/onemodels/mace/mace-mp-0-medium-portable.pt')
        model_path = Path(MODEL_PATH).expanduser().resolve()
        standard_path = (REFERENCE_DIR / 'mace_standalone_inference.py').resolve()
        spec = importlib.util.spec_from_file_location('mace_standalone_inference_standard', standard_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load module: {standard_path}')
        standard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standard)
        if os.environ.get('MACE_STANDALONE_STANDARD_ONLY') == '1':
            candidate = standard
        else:
            SUBMISSION_PATH = (CANDIDATE_DIR / 'mace_standalone_inference.py').resolve()
            spec = importlib.util.spec_from_file_location('mace_standalone_inference_candidate', SUBMISSION_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f'cannot load module: {SUBMISSION_PATH}')
            candidate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate)
        device = os.environ.get('MACE_STANDALONE_EVAL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        with tempfile.TemporaryDirectory(prefix='mace-standalone-inference-') as temporary:
            root = Path(temporary)
            water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]], dtype=np.float64)
            frames = [Atoms(numbers=[8, 1, 1], positions=water + np.array([[0.0, 0.0, 0.0], [delta, 0.0, 0.0], [0.0, -0.5 * delta, 0.0]]), cell=[10.0, 10.0, 10.0], pbc=False) for delta in (0.0, 0.04)]
            frames.append(Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]], cell=[12.0, 12.0, 12.0], pbc=[True, False, False]))
            ase_file = root / 'structures.extxyz'
            write(ase_file, frames, format='extxyz')
            hdf5_file = root / 'structures.h5'
            hdf5_dir = root / 'structures_hdf5'
            hdf5_dir.mkdir()
            for hdf5_path, hdf5_frames in ((hdf5_file, frames), (hdf5_dir / 'shard_0.h5', frames[:2]), (hdf5_dir / 'shard_1.hdf5', frames[2:])):
                with h5py.File(hdf5_path, 'w') as handle:
                    batch = handle.create_group('config_batch_0')
                    for index, atoms in enumerate(hdf5_frames):
                        group = batch.create_group(f'config_{index}')
                        group['atomic_numbers'] = atoms.get_atomic_numbers()
                        group['positions'] = atoms.get_positions()
                        group['cell'] = atoms.cell.array
                        group['pbc'] = atoms.pbc
            standard_output = root / 'standard'
            standard.MACEStandaloneInferenceRunner(model_path=str(model_path), input_path=ase_file, output_dir=standard_output, data_format='ase', device=device).run()
            update_check(current_index, True, 'standard ASE inference completed')
            current_index += 1
            cases = [('ase_file', ase_file, 'ase', len(frames)), ('ase_single_atoms', frames[0], 'ase', 1), ('ase_atoms_sequence', frames, 'ase', len(frames)), ('hdf5_file', hdf5_file, 'hdf5', len(frames)), ('hdf5_sharded_dir', hdf5_dir, 'hdf5', len(frames))]
            for case_name, input_path, data_format, count in cases:
                output_dir = root / f'candidate_{case_name}'
                candidate.MACEStandaloneInferenceRunner(model_path=str(model_path), input_path=input_path, output_dir=output_dir, data_format=data_format, device=device).run()
                update_check(current_index, True, f'{case_name} inference completed for {count} structures')
                current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_output_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'ase_file_output_files', 'ase_file_manifest_model_path', 'ase_file_manifest_data_format', 'ase_file_manifest_count', 'ase_file_record_count', 'ase_file_record_0_index', 'ase_file_record_0_force_shape', 'ase_file_record_0_finite_values', 'ase_file_record_1_index', 'ase_file_record_1_force_shape', 'ase_file_record_1_finite_values', 'ase_file_record_2_index', 'ase_file_record_2_force_shape', 'ase_file_record_2_finite_values',
        'ase_single_atoms_output_files', 'ase_single_atoms_manifest_model_path', 'ase_single_atoms_manifest_data_format', 'ase_single_atoms_manifest_count', 'ase_single_atoms_record_count', 'ase_single_atoms_record_0_index', 'ase_single_atoms_record_0_force_shape', 'ase_single_atoms_record_0_finite_values',
        'ase_atoms_sequence_output_files', 'ase_atoms_sequence_manifest_model_path', 'ase_atoms_sequence_manifest_data_format', 'ase_atoms_sequence_manifest_count', 'ase_atoms_sequence_record_count', 'ase_atoms_sequence_record_0_index', 'ase_atoms_sequence_record_0_force_shape', 'ase_atoms_sequence_record_0_finite_values', 'ase_atoms_sequence_record_1_index', 'ase_atoms_sequence_record_1_force_shape', 'ase_atoms_sequence_record_1_finite_values', 'ase_atoms_sequence_record_2_index', 'ase_atoms_sequence_record_2_force_shape', 'ase_atoms_sequence_record_2_finite_values',
        'hdf5_file_output_files', 'hdf5_file_manifest_model_path', 'hdf5_file_manifest_data_format', 'hdf5_file_manifest_count', 'hdf5_file_record_count', 'hdf5_file_record_0_index', 'hdf5_file_record_0_force_shape', 'hdf5_file_record_0_finite_values', 'hdf5_file_record_1_index', 'hdf5_file_record_1_force_shape', 'hdf5_file_record_1_finite_values', 'hdf5_file_record_2_index', 'hdf5_file_record_2_force_shape', 'hdf5_file_record_2_finite_values',
        'hdf5_sharded_dir_output_files', 'hdf5_sharded_dir_manifest_model_path', 'hdf5_sharded_dir_manifest_data_format', 'hdf5_sharded_dir_manifest_count', 'hdf5_sharded_dir_record_count', 'hdf5_sharded_dir_record_0_index', 'hdf5_sharded_dir_record_0_force_shape', 'hdf5_sharded_dir_record_0_finite_values', 'hdf5_sharded_dir_record_1_index', 'hdf5_sharded_dir_record_1_force_shape', 'hdf5_sharded_dir_record_1_finite_values', 'hdf5_sharded_dir_record_2_index', 'hdf5_sharded_dir_record_2_force_shape', 'hdf5_sharded_dir_record_2_finite_values',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        MODEL_PATH = os.environ.get('MACE_STANDALONE_MODEL_PATH', '/public/share/sugonhpcapp01/onestore/onemodels/mace/mace-mp-0-medium-portable.pt')
        model_path = Path(MODEL_PATH).expanduser().resolve()
        if os.environ.get('MACE_STANDALONE_STANDARD_ONLY') == '1':
            standard_path = (REFERENCE_DIR / 'mace_standalone_inference.py').resolve()
            spec = importlib.util.spec_from_file_location('mace_standalone_inference_standard', standard_path)
        else:
            SUBMISSION_PATH = (CANDIDATE_DIR / 'mace_standalone_inference.py').resolve()
            spec = importlib.util.spec_from_file_location('mace_standalone_inference_candidate', SUBMISSION_PATH)
        if spec is None or spec.loader is None:
            raise ImportError('cannot load candidate module')
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        device = os.environ.get('MACE_STANDALONE_EVAL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        with tempfile.TemporaryDirectory(prefix='mace-standalone-inference-') as temporary:
            root = Path(temporary)
            water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]], dtype=np.float64)
            frames = [Atoms(numbers=[8, 1, 1], positions=water + np.array([[0.0, 0.0, 0.0], [delta, 0.0, 0.0], [0.0, -0.5 * delta, 0.0]]), cell=[10.0, 10.0, 10.0], pbc=False) for delta in (0.0, 0.04)]
            frames.append(Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]], cell=[12.0, 12.0, 12.0], pbc=[True, False, False]))
            ase_file = root / 'structures.extxyz'
            write(ase_file, frames, format='extxyz')
            hdf5_file = root / 'structures.h5'
            hdf5_dir = root / 'structures_hdf5'
            hdf5_dir.mkdir()
            for hdf5_path, hdf5_frames in ((hdf5_file, frames), (hdf5_dir / 'shard_0.h5', frames[:2]), (hdf5_dir / 'shard_1.hdf5', frames[2:])):
                with h5py.File(hdf5_path, 'w') as handle:
                    batch = handle.create_group('config_batch_0')
                    for index, atoms in enumerate(hdf5_frames):
                        group = batch.create_group(f'config_{index}')
                        group['atomic_numbers'] = atoms.get_atomic_numbers()
                        group['positions'] = atoms.get_positions()
                        group['cell'] = atoms.cell.array
                        group['pbc'] = atoms.pbc
            cases = [('ase_file', ase_file, 'ase', len(frames)), ('ase_single_atoms', frames[0], 'ase', 1), ('ase_atoms_sequence', frames, 'ase', len(frames)), ('hdf5_file', hdf5_file, 'hdf5', len(frames)), ('hdf5_sharded_dir', hdf5_dir, 'hdf5', len(frames))]
            for case_name, input_path, data_format, expected_count in cases:
                output_dir = root / f'candidate_{case_name}'
                candidate.MACEStandaloneInferenceRunner(model_path=str(model_path), input_path=input_path, output_dir=output_dir, data_format=data_format, device=device).run()
                manifest_path = output_dir / 'inference_manifest.json'
                predictions_path = output_dir / 'predictions.jsonl'
                assert manifest_path.is_file() and predictions_path.is_file(), 'required outputs are missing'
                update_check(current_index, True, f'{case_name} required outputs exist')
                current_index += 1
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                assert Path(manifest['model_path']).expanduser().resolve() == Path(MODEL_PATH).resolve()
                update_check(current_index, True, f'{case_name} manifest model path is correct')
                current_index += 1
                assert manifest['data_format'] == data_format
                update_check(current_index, True, f'{case_name} manifest data format is correct')
                current_index += 1
                assert manifest['num_structures'] == expected_count
                update_check(current_index, True, f'{case_name} manifest structure count is correct')
                current_index += 1
                records = [json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                assert len(records) == expected_count
                update_check(current_index, True, f'{case_name} prediction record count is correct')
                current_index += 1
                for index, record in enumerate(records):
                    assert record['index'] == index
                    update_check(current_index, True, f'{case_name} record {index} index is correct')
                    current_index += 1
                    numbers = record['atomic_numbers']
                    forces = np.asarray(record['forces'], dtype=np.float64)
                    assert forces.shape == (len(numbers), 3)
                    update_check(current_index, True, f'{case_name} record {index} force shape is correct')
                    current_index += 1
                    assert np.isfinite(float(record['energy'])) and np.isfinite(forces).all()
                    update_check(current_index, True, f'{case_name} record {index} values are finite')
                    current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'ase_file_record_count', 'ase_file_record_0_index', 'ase_file_record_0_atomic_numbers', 'ase_file_record_1_index', 'ase_file_record_1_atomic_numbers', 'ase_file_record_2_index', 'ase_file_record_2_atomic_numbers', 'ase_file_energy_error', 'ase_file_force_error',
        'ase_single_atoms_record_count', 'ase_single_atoms_record_0_index', 'ase_single_atoms_record_0_atomic_numbers', 'ase_single_atoms_energy_error', 'ase_single_atoms_force_error',
        'ase_atoms_sequence_record_count', 'ase_atoms_sequence_record_0_index', 'ase_atoms_sequence_record_0_atomic_numbers', 'ase_atoms_sequence_record_1_index', 'ase_atoms_sequence_record_1_atomic_numbers', 'ase_atoms_sequence_record_2_index', 'ase_atoms_sequence_record_2_atomic_numbers', 'ase_atoms_sequence_energy_error', 'ase_atoms_sequence_force_error',
        'hdf5_file_record_count', 'hdf5_file_record_0_index', 'hdf5_file_record_0_atomic_numbers', 'hdf5_file_record_1_index', 'hdf5_file_record_1_atomic_numbers', 'hdf5_file_record_2_index', 'hdf5_file_record_2_atomic_numbers', 'hdf5_file_energy_error', 'hdf5_file_force_error',
        'hdf5_sharded_dir_record_count', 'hdf5_sharded_dir_record_0_index', 'hdf5_sharded_dir_record_0_atomic_numbers', 'hdf5_sharded_dir_record_1_index', 'hdf5_sharded_dir_record_1_atomic_numbers', 'hdf5_sharded_dir_record_2_index', 'hdf5_sharded_dir_record_2_atomic_numbers', 'hdf5_sharded_dir_energy_error', 'hdf5_sharded_dir_force_error',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        MODEL_PATH = os.environ.get('MACE_STANDALONE_MODEL_PATH', '/public/share/sugonhpcapp01/onestore/onemodels/mace/mace-mp-0-medium-portable.pt')
        model_path = Path(MODEL_PATH).expanduser().resolve()
        standard_path = (REFERENCE_DIR / 'mace_standalone_inference.py').resolve()
        spec = importlib.util.spec_from_file_location('mace_standalone_inference_standard', standard_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load module: {standard_path}')
        standard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standard)
        if os.environ.get('MACE_STANDALONE_STANDARD_ONLY') == '1':
            candidate = standard
        else:
            SUBMISSION_PATH = (CANDIDATE_DIR / 'mace_standalone_inference.py').resolve()
            spec = importlib.util.spec_from_file_location('mace_standalone_inference_candidate', SUBMISSION_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f'cannot load module: {SUBMISSION_PATH}')
            candidate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate)
        device = os.environ.get('MACE_STANDALONE_EVAL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        with tempfile.TemporaryDirectory(prefix='mace-standalone-inference-') as temporary:
            root = Path(temporary)
            water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]], dtype=np.float64)
            frames = [Atoms(numbers=[8, 1, 1], positions=water + np.array([[0.0, 0.0, 0.0], [delta, 0.0, 0.0], [0.0, -0.5 * delta, 0.0]]), cell=[10.0, 10.0, 10.0], pbc=False) for delta in (0.0, 0.04)]
            frames.append(Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]], cell=[12.0, 12.0, 12.0], pbc=[True, False, False]))
            ase_file = root / 'structures.extxyz'
            write(ase_file, frames, format='extxyz')
            hdf5_file = root / 'structures.h5'
            hdf5_dir = root / 'structures_hdf5'
            hdf5_dir.mkdir()
            for hdf5_path, hdf5_frames in ((hdf5_file, frames), (hdf5_dir / 'shard_0.h5', frames[:2]), (hdf5_dir / 'shard_1.hdf5', frames[2:])):
                with h5py.File(hdf5_path, 'w') as handle:
                    batch = handle.create_group('config_batch_0')
                    for index, atoms in enumerate(hdf5_frames):
                        group = batch.create_group(f'config_{index}')
                        group['atomic_numbers'] = atoms.get_atomic_numbers()
                        group['positions'] = atoms.get_positions()
                        group['cell'] = atoms.cell.array
                        group['pbc'] = atoms.pbc
            standard_output = root / 'standard'
            standard.MACEStandaloneInferenceRunner(model_path=str(model_path), input_path=ase_file, output_dir=standard_output, data_format='ase', device=device).run()
            manifest_path = standard_output / 'inference_manifest.json'
            predictions_path = standard_output / 'predictions.jsonl'
            assert manifest_path.is_file() and predictions_path.is_file(), 'required outputs are missing'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            assert Path(manifest['model_path']).expanduser().resolve() == Path(MODEL_PATH).resolve()
            assert manifest['data_format'] == 'ase'
            assert manifest['num_structures'] == len(frames)
            reference = [json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            assert len(reference) == len(frames)
            for index, record in enumerate(reference):
                assert record['index'] == index
                numbers = record['atomic_numbers']
                forces = np.asarray(record['forces'], dtype=np.float64)
                assert forces.shape == (len(numbers), 3)
                assert np.isfinite(float(record['energy'])) and np.isfinite(forces).all()
            cases = [('ase_file', ase_file, 'ase', len(frames), reference), ('ase_single_atoms', frames[0], 'ase', 1, reference[:1]), ('ase_atoms_sequence', frames, 'ase', len(frames), reference), ('hdf5_file', hdf5_file, 'hdf5', len(frames), reference), ('hdf5_sharded_dir', hdf5_dir, 'hdf5', len(frames), reference)]
            for case_name, input_path, data_format, count, expected in cases:
                output_dir = root / f'candidate_{case_name}'
                candidate.MACEStandaloneInferenceRunner(model_path=str(model_path), input_path=input_path, output_dir=output_dir, data_format=data_format, device=device).run()
                manifest_path = output_dir / 'inference_manifest.json'
                predictions_path = output_dir / 'predictions.jsonl'
                assert manifest_path.is_file() and predictions_path.is_file(), 'required outputs are missing'
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                assert Path(manifest['model_path']).expanduser().resolve() == Path(MODEL_PATH).resolve()
                assert manifest['data_format'] == data_format
                assert manifest['num_structures'] == count
                observed = [json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                assert len(observed) == count
                for index, record in enumerate(observed):
                    assert record['index'] == index
                    numbers = record['atomic_numbers']
                    forces = np.asarray(record['forces'], dtype=np.float64)
                    assert forces.shape == (len(numbers), 3)
                    assert np.isfinite(float(record['energy'])) and np.isfinite(forces).all()
                assert len(expected) == len(observed)
                update_check(current_index, True, f'{case_name} record counts match')
                current_index += 1
                max_energy_error = 0.0
                max_force_error = 0.0
                for expected_record, actual in zip(expected, observed):
                    assert expected_record['index'] == actual['index']
                    update_check(current_index, True, f'{case_name} record index matches')
                    current_index += 1
                    assert expected_record['atomic_numbers'] == actual['atomic_numbers']
                    update_check(current_index, True, f'{case_name} atomic numbers match')
                    current_index += 1
                    max_energy_error = max(max_energy_error, abs(float(expected_record['energy']) - float(actual['energy'])))
                    max_force_error = max(max_force_error, float(np.max(np.abs(np.asarray(expected_record['forces'], dtype=np.float64) - np.asarray(actual['forces'], dtype=np.float64)))))
                assert max_energy_error < 1.0e-6
                update_check(current_index, True, f'{case_name} maximum energy error is {max_energy_error}')
                current_index += 1
                assert max_force_error < 1.0e-6
                update_check(current_index, True, f'{case_name} maximum force error is {max_force_error}')
                current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks
