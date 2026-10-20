'''V2 evaluator adapter for autoregressive_rollout_infer.'''
from pathlib import Path
import sys
import importlib.util
import inspect
import os
import traceback
from typing import Dict

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

REQUIRED_FILES = ('autoregressive_rollout_infer.py',)
REQUIRED_CLASS_NAME = 'FourCastNetRollout'
MODEL_CONFIG = {
    'img_size': (32, 64),
    'patch_size': (4, 4),
    'in_chans': 20,
    'out_chans': 20,
    'embed_dim': 256,
    'depth': 4,
    'num_blocks': 4,
}
FORWARD_CASES = ((1, 123), (2, 456))
ROLLOUT_STEPS = 3
FORWARD_COMPARE_ATOL = 1e-5
FORWARD_COMPARE_RTOL = 1e-5


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:autoregressive_rollout_infer.py']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_file = (CANDIDATE_DIR / REQUIRED_FILES[0]).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        ok = candidate_file.is_file()
        update_check(current_index, ok, str(candidate_file))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['checkpoint:not_required']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        detail = 'not applicable: this task defines a model from scratch with no pre-trained checkpoint'
        update_check(current_index, True, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'eval:forward_exists',
        'eval:forward_signature_ok',
        'eval:autoregressive_rollout_exists',
        'eval:rollout_signature_ok',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import inspect

        model_file = config.get('model_file', REQUIRED_FILES[0])
        model_config = dict(config.get('model_config', MODEL_CONFIG))
        candidate_model_path = (CANDIDATE_DIR / model_file).resolve()
        candidate_model_path.relative_to(CANDIDATE_DIR)
        directory = str(CANDIDATE_DIR)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        spec = importlib.util.spec_from_file_location(
            'autoregressive_rollout_submission_model', str(candidate_model_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {candidate_model_path}')
        module_dir = str(candidate_model_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules['autoregressive_rollout_submission_model'] = model_module
        spec.loader.exec_module(model_module)
        model_cls = getattr(model_module, config.get('model_class', REQUIRED_CLASS_NAME), None)
        model = model_cls(**model_config).cpu().eval()

        forward_method = getattr(model, 'forward', None)
        update_check(
            current_index,
            callable(forward_method),
            str(type(forward_method)),
        )
        current_index += 1
        if callable(forward_method):
            forward_signature = inspect.signature(forward_method)
            update_check(
                current_index,
                'x' in forward_signature.parameters,
                str(forward_signature),
            )
        else:
            update_check(current_index, False, 'forward is not callable')
        current_index += 1

        rollout_method = getattr(model, 'autoregressive_rollout', None)
        update_check(
            current_index,
            callable(rollout_method),
            str(type(rollout_method)),
        )
        current_index += 1
        if callable(rollout_method):
            rollout_sig = inspect.signature(rollout_method)
            update_check(
                current_index,
                all(k in rollout_sig.parameters for k in ('initial_state', 'num_steps')),
                str(rollout_sig),
            )
        else:
            update_check(current_index, False, 'autoregressive_rollout is not callable')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'model:init_supports_training_config',
        'model:is_nn_module',
        'model:has_trainable_parameters',
        'structure:patch_embedding',
        'structure:position_embedding',
        'model:has_block_stack',
        'model:top_level_order_ok',
        'structure:patch_recovery_head',
        'structure:afno_frequency_mixing',
        'structure:pre_norm_blocks',
        'structure:block_shapes_preserved',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        import torch.nn as nn

        model_file = config.get('model_file', REQUIRED_FILES[0])
        model_config = dict(config.get('model_config', MODEL_CONFIG))
        candidate_model_path = (CANDIDATE_DIR / model_file).resolve()
        candidate_model_path.relative_to(CANDIDATE_DIR)
        directory = str(CANDIDATE_DIR)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        spec = importlib.util.spec_from_file_location(
            'autoregressive_rollout_submission_model', str(candidate_model_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {candidate_model_path}')
        module_dir = str(candidate_model_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['autoregressive_rollout_submission_model'] = train_module
        spec.loader.exec_module(train_module)
        model_cls = getattr(train_module, config.get('model_class', REQUIRED_CLASS_NAME), None)

        init_signature = inspect.signature(model_cls.__init__)
        required_init_keys = (
            'img_size',
            'patch_size',
            'in_chans',
            'out_chans',
            'embed_dim',
            'depth',
            'num_blocks',
        )
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in init_signature.parameters.values()
        )
        missing_keys = [
            k for k in required_init_keys
            if k not in init_signature.parameters and not accepts_kwargs
        ]
        update_check(
            current_index,
            not missing_keys,
            f'signature={init_signature}, missing={missing_keys}',
        )
        current_index += 1

        model = model_cls(**model_config).cpu().eval()
        update_check(current_index, isinstance(model, nn.Module), str(type(model)))
        current_index += 1

        parameter_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        update_check(
            current_index,
            parameter_count > 0 and trainable_count > 0,
            f'parameters={parameter_count}, trainable={trainable_count}',
        )
        current_index += 1

        modules = list(model.modules())
        patch_layers = [
            m
            for m in modules
            if isinstance(m, nn.Conv2d)
            and m.in_channels == MODEL_CONFIG['in_chans']
            and m.out_channels == MODEL_CONFIG['embed_dim']
            and tuple(m.kernel_size) == MODEL_CONFIG['patch_size']
            and tuple(m.stride) == MODEL_CONFIG['patch_size']
        ]
        update_check(
            current_index,
            bool(patch_layers),
            'Conv2d(20, 256, kernel_size=(4,4), stride=(4,4))',
        )
        current_index += 1

        h_patches = MODEL_CONFIG['img_size'][0] // MODEL_CONFIG['patch_size'][0]
        w_patches = MODEL_CONFIG['img_size'][1] // MODEL_CONFIG['patch_size'][1]
        expected_pos_shape = (1, h_patches * w_patches, MODEL_CONFIG['embed_dim'])
        pos_params = [
            (name, p)
            for name, p in model.named_parameters()
            if tuple(p.shape) == expected_pos_shape
        ]
        update_check(
            current_index,
            bool(pos_params),
            f'expected_shape={expected_pos_shape}, matches={[name for name, _ in pos_params]}',
        )
        current_index += 1

        block_stacks = [
            m
            for m in modules
            if isinstance(m, (nn.ModuleList, nn.Sequential))
            and len(m) == MODEL_CONFIG['depth']
        ]
        update_check(
            current_index,
            bool(block_stacks),
            f'depth={MODEL_CONFIG["depth"]}',
        )
        current_index += 1

        if block_stacks and patch_layers:
            execution_order = []
            handles = [patch_layers[0].register_forward_hook(lambda *_: execution_order.append('patch'))]
            handles.extend(block.register_forward_hook(lambda *_: execution_order.append('block')) for block in block_stacks[0])
            try:
                with torch.no_grad():
                    model(torch.randn(1, MODEL_CONFIG['in_chans'], *MODEL_CONFIG['img_size']))
                patch_idx = execution_order.index('patch')
                block_idx = execution_order.index('block')
                order_ok = patch_idx < block_idx and execution_order.count('block') == MODEL_CONFIG['depth']
                detail = str(execution_order)
            except Exception as exc:
                order_ok = False
                detail = f'{type(exc).__name__}: {exc}'
            finally:
                for handle in handles:
                    handle.remove()
            update_check(current_index, order_ok, detail)
        else:
            update_check(current_index, False, 'patch embedding or block stack missing')
        current_index += 1

        expected_head_features = (
            MODEL_CONFIG['out_chans']
            * MODEL_CONFIG['patch_size'][0]
            * MODEL_CONFIG['patch_size'][1]
        )
        recovery_heads = [
            m
            for m in modules
            if isinstance(m, nn.Linear)
            and m.in_features == MODEL_CONFIG['embed_dim']
            and m.out_features == expected_head_features
        ]
        update_check(
            current_index,
            bool(recovery_heads),
            f'Linear({MODEL_CONFIG["embed_dim"]}, {expected_head_features})',
        )
        current_index += 1

        source_path = candidate_model_path.resolve()
        source_text = source_path.read_text(encoding='utf-8')
        normalized_source = ''.join(source_text.lower().split())
        has_forward_fft = 'rfft2(' in normalized_source
        has_inverse_fft = 'irfft2(' in normalized_source
        update_check(
            current_index,
            has_forward_fft and has_inverse_fft,
            f'rfft2={has_forward_fft}, irfft2={has_inverse_fft}',
        )
        current_index += 1

        layer_norm_count = sum(isinstance(m, nn.LayerNorm) for m in modules)
        update_check(
            current_index,
            layer_norm_count >= 2 * MODEL_CONFIG['depth'],
            f'layer_norm_count={layer_norm_count}',
        )
        current_index += 1

        sample = torch.randn(2, h_patches, w_patches, MODEL_CONFIG['embed_dim'])
        shape_preserving = 0
        with torch.no_grad():
            for stack in block_stacks[:1]:
                for block in stack:
                    try:
                        block_out = block(sample)
                    except Exception:
                        continue
                    if isinstance(block_out, torch.Tensor) and tuple(
                        block_out.shape
                    ) == tuple(sample.shape):
                        shape_preserving += 1
        update_check(
            current_index,
            shape_preserving == MODEL_CONFIG['depth'],
            f'shape_preserving_blocks={shape_preserving}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'forward:b1_seed123_returns_tensor',
        'forward:b1_seed123_shape',
        'forward:b1_seed123_finite',
        'forward:b1_seed123_deterministic',
        'forward:b1_seed123_input_unchanged',
        'forward:b2_seed456_returns_tensor',
        'forward:b2_seed456_shape',
        'forward:b2_seed456_finite',
        'forward:b2_seed456_deterministic',
        'forward:b2_seed456_input_unchanged',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        model_file = config.get('model_file', REQUIRED_FILES[0])
        model_config = dict(config.get('model_config', MODEL_CONFIG))
        candidate_model_path = (CANDIDATE_DIR / model_file).resolve()
        candidate_model_path.relative_to(CANDIDATE_DIR)
        directory = str(CANDIDATE_DIR)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        spec = importlib.util.spec_from_file_location(
            'autoregressive_rollout_submission_model', str(candidate_model_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {candidate_model_path}')
        module_dir = str(candidate_model_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['autoregressive_rollout_submission_model'] = train_module
        spec.loader.exec_module(train_module)
        model_cls = getattr(train_module, config.get('model_class', REQUIRED_CLASS_NAME), None)
        model = model_cls(**model_config).to('cpu').eval()
        device = 'cpu'

        for batch_size, seed in FORWARD_CASES:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed)
            x = torch.randn(
                batch_size,
                MODEL_CONFIG['in_chans'],
                *MODEL_CONFIG['img_size'],
                generator=generator,
            ).to(device)
            original_x = x.clone()
            prefix = f'forward:b{batch_size}_seed{seed}'
            try:
                with torch.no_grad():
                    first = model(x)
                    second = model(x)
            except Exception as exc:
                detail = f'{type(exc).__name__}: {exc}'
                update_check(current_index, False, detail)
                current_index += 1
                update_check(current_index, False, detail)
                current_index += 1
                update_check(current_index, False, detail)
                current_index += 1
                update_check(current_index, False, detail)
                current_index += 1
                update_check(current_index, False, detail)
                current_index += 1
                continue

            returns_tensor = isinstance(first, torch.Tensor)
            update_check(current_index, returns_tensor, str(type(first)))
            current_index += 1
            if not returns_tensor:
                detail = f'forward returned {type(first)}'
                update_check(current_index, False, detail)
                current_index += 1
                update_check(current_index, False, detail)
                current_index += 1
                update_check(current_index, False, detail)
                current_index += 1
                update_check(current_index, False, detail)
                current_index += 1
                continue

            expected_shape = (
                batch_size,
                MODEL_CONFIG['out_chans'],
                *MODEL_CONFIG['img_size'],
            )
            update_check(
                current_index,
                tuple(first.shape) == expected_shape,
                f'expected={expected_shape}, actual={tuple(first.shape)}',
            )
            current_index += 1
            update_check(
                current_index,
                bool(torch.isfinite(first).all()),
                'all finite',
            )
            current_index += 1
            update_check(
                current_index,
                bool(torch.equal(first, second)),
                'repeated eval-mode forward',
            )
            current_index += 1
            update_check(
                current_index,
                bool(torch.equal(x, original_x)),
                'forward does not mutate input',
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'rollout:returns_tensor',
        'rollout:trajectory_shape',
        'rollout:finite',
        'rollout:no_gradient_graph',
        'rollout:deterministic',
        'rollout:matches_iterative_forward',
        'rollout:states_evolve',
        'rollout:input_unchanged',
        'rollout:one_step_boundary',
        'boundary:num_steps_zero_raises_value_error',
        'boundary:num_steps_negative_raises_value_error',
        'boundary:num_steps_float_raises_value_error',
        'boundary:num_steps_bool_raises_value_error',
        'boundary:wrong_channels_rejected',
        'boundary:wrong_spatial_size_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        model_file = config.get('model_file', REQUIRED_FILES[0])
        model_config = dict(config.get('model_config', MODEL_CONFIG))
        candidate_model_path = (CANDIDATE_DIR / model_file).resolve()
        candidate_model_path.relative_to(CANDIDATE_DIR)
        directory = str(CANDIDATE_DIR)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        spec = importlib.util.spec_from_file_location(
            'autoregressive_rollout_submission_model', str(candidate_model_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {candidate_model_path}')
        module_dir = str(candidate_model_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['autoregressive_rollout_submission_model'] = train_module
        spec.loader.exec_module(train_module)
        model_cls = getattr(train_module, config.get('model_class', REQUIRED_CLASS_NAME), None)

        generated_model = model_cls(**model_config).to('cpu').eval()
        device = 'cpu'
        generator = torch.Generator(device='cpu')
        generator.manual_seed(789)
        x = torch.randn(
            1,
            MODEL_CONFIG['in_chans'],
            *MODEL_CONFIG['img_size'],
            generator=generator,
        ).to(device)
        original_x = x.clone()
        trajectory = generated_model.autoregressive_rollout(x, ROLLOUT_STEPS)
        repeated_trajectory = generated_model.autoregressive_rollout(x, ROLLOUT_STEPS)

        expected_shape = (
            ROLLOUT_STEPS,
            1,
            MODEL_CONFIG['out_chans'],
            *MODEL_CONFIG['img_size'],
        )
        returns_tensor = isinstance(trajectory, torch.Tensor)
        update_check(current_index, returns_tensor, str(type(trajectory)))
        current_index += 1
        if returns_tensor:
            update_check(
                current_index,
                tuple(trajectory.shape) == expected_shape,
                f'expected={expected_shape}, actual={tuple(trajectory.shape)}',
            )
            current_index += 1
            update_check(
                current_index,
                bool(torch.isfinite(trajectory).all()),
                'all finite',
            )
            current_index += 1
            update_check(
                current_index,
                not trajectory.requires_grad and trajectory.grad_fn is None,
                f'requires_grad={trajectory.requires_grad}, grad_fn={trajectory.grad_fn}',
            )
            current_index += 1
            update_check(
                current_index,
                bool(torch.equal(trajectory, repeated_trajectory)),
                'repeated eval-mode rollout',
            )
            current_index += 1

            current = x
            manual_steps = []
            with torch.no_grad():
                for _ in range(ROLLOUT_STEPS):
                    current = generated_model(current)
                    manual_steps.append(current.clone())
                manual_trajectory = torch.stack(manual_steps, dim=0)
            manual_match = bool(
                torch.allclose(
                    trajectory,
                    manual_trajectory,
                    atol=FORWARD_COMPARE_ATOL,
                    rtol=FORWARD_COMPARE_RTOL,
                )
            )
            update_check(
                current_index,
                manual_match,
                'each prediction is fed into the next step',
            )
            current_index += 1
            update_check(
                current_index,
                not bool(torch.equal(trajectory[0], trajectory[-1])),
                'first and final predictions differ',
            )
            current_index += 1
        else:
            detail = f'rollout returned {type(trajectory)}'
            update_check(current_index, False, detail)
            current_index += 1
            update_check(current_index, False, detail)
            current_index += 1
            update_check(current_index, False, detail)
            current_index += 1
            update_check(current_index, False, detail)
            current_index += 1
            update_check(current_index, False, detail)
            current_index += 1
            update_check(current_index, False, detail)
            current_index += 1

        update_check(
            current_index,
            bool(torch.equal(x, original_x)),
            'rollout does not mutate initial_state',
        )
        current_index += 1

        try:
            one_step = generated_model.autoregressive_rollout(x, 1)
            with torch.no_grad():
                expected_one_step = generated_model(x).unsqueeze(0)
            one_step_ok = (
                isinstance(one_step, torch.Tensor)
                and one_step.shape == expected_one_step.shape
                and bool(
                    torch.allclose(
                        one_step,
                        expected_one_step,
                        atol=FORWARD_COMPARE_ATOL,
                        rtol=FORWARD_COMPARE_RTOL,
                    )
                )
            )
            one_step_detail = f'shape={tuple(one_step.shape) if isinstance(one_step, torch.Tensor) else None}'
        except Exception as exc:
            one_step_ok = False
            one_step_detail = f'{type(exc).__name__}: {exc}'
        update_check(current_index, one_step_ok, one_step_detail)
        current_index += 1

        model = model_cls(**model_config).to(device).eval()
        generator = torch.Generator(device='cpu')
        generator.manual_seed(101)
        valid_x = torch.randn(
            1,
            MODEL_CONFIG['in_chans'],
            *MODEL_CONFIG['img_size'],
            generator=generator,
        ).to(device)

        for label, num_steps in (('zero', 0), ('negative', -1), ('float', 1.5), ('bool', True)):
            try:
                model.autoregressive_rollout(valid_x, num_steps)
            except ValueError:
                ok = True
                detail = f'ValueError raised for num_steps={num_steps!r}'
            except Exception as exc:
                ok = False
                detail = f'wrong exception {type(exc).__name__}: {exc}'
            else:
                ok = False
                detail = 'no exception raised'
            update_check(current_index, ok, detail)
            current_index += 1

        invalid_batches = {
            'wrong_channels': torch.randn(1, 19, 32, 64, device=device),
            'wrong_spatial_size': torch.randn(1, 20, 28, 64, device=device),
        }
        for case_name, invalid_x in invalid_batches.items():
            try:
                with torch.no_grad():
                    model(invalid_x)
            except Exception as exc:
                ok = True
                detail = f'{type(exc).__name__}: {exc}'
            else:
                ok = False
                detail = 'invalid input was accepted'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
