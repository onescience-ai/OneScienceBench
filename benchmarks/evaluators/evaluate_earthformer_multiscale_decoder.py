"""V2 adapter for the Earthformer multiscale decoder evaluator."""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path
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

REQUIRED_FILES = ("earthformer_multiscale_decoder.py", "train.py")
ENV_NAME = "EARTHFORMER_MULTISCALE_DECODER_SUBMISSION_DIR"
HINTS = ("earthformer_multiscale_decoder", "earthformer_multiscale_decoder_V1")
ALLOWED = {"file", "data", "train_script", "model_structure", "forward_executable", "forward_equivalence", "traning"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["file:earthformer_multiscale_decoder.py", "file:train.py"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        required_files = tuple(config.get("legacy_required_files", REQUIRED_FILES))
        for name in required_files:
            candidate_file = (CANDIDATE_DIR / name).resolve()
            update_check(current_index, candidate_file.is_file(), str(candidate_file))
            current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["data:multiscale_feature_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn
        module = load_module("earthformer_multiscale_decoder_submission", (CANDIDATE_DIR / REQUIRED_FILES[0]).resolve())
        train = load_module("earthformer_multiscale_decoder_train_submission", (CANDIDATE_DIR / REQUIRED_FILES[1]).resolve())
        model_cls = getattr(module, "EarthformerMultiscaleDecoder")
        fusion_cls = getattr(module, "MultiscaleFusionBlock")
        model = train.build_model({"encoder_channels": [8, 16, 32], "hidden_channels": 16, "out_channels": 2})
        features = (torch.randn(2, 8, 3, 8, 8), torch.randn(2, 16, 3, 4, 4), torch.randn(2, 32, 3, 2, 2))
        target = torch.randn(2, 3, 2, 8, 8)
        output = model(features)
        update_check(
            current_index,
            all(t.shape[0] == 2 and t.shape[2] == 3 for t in features) and target.shape == (2, 3, 2, 8, 8),
            "features=[shallow,middle,deep] in [B,C,T,H,W]",
        )
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["train_script:public_entrypoints"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        files = config.get("files", ("train.py",))
        required_interfaces = config.get("required_interfaces", ("build_model", "train_step"))
        train_path = (CANDIDATE_DIR / files[0]).resolve()
        train = load_module("earthformer_multiscale_decoder_train_submission", train_path)
        update_check(
            current_index,
            callable(getattr(train, required_interfaces[0], None)) and callable(getattr(train, required_interfaces[1], None)),
            "build_model/train_step",
        )
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_equivalence:deep_path_influences_output"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn
        module = load_module("earthformer_multiscale_decoder_submission", (CANDIDATE_DIR / REQUIRED_FILES[0]).resolve())
        train = load_module("earthformer_multiscale_decoder_train_submission", (CANDIDATE_DIR / REQUIRED_FILES[1]).resolve())
        model_cls = getattr(module, "EarthformerMultiscaleDecoder")
        fusion_cls = getattr(module, "MultiscaleFusionBlock")
        model = train.build_model({"encoder_channels": [8, 16, 32], "hidden_channels": 16, "out_channels": 2})
        features = (torch.randn(2, 8, 3, 8, 8), torch.randn(2, 16, 3, 4, 4), torch.randn(2, 32, 3, 2, 2))
        target = torch.randn(2, 3, 2, 8, 8)
        output = model(features)
        altered = list(features)
        altered[2] = altered[2] + 2.0
        output_altered = model(tuple(altered))
        update_check(
            current_index,
            not bool(torch.allclose(output, output_altered)),
            "deep feature reaches the final prediction through top-down fusion",
        )
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_executable:shape_and_finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn
        module = load_module("earthformer_multiscale_decoder_submission", (CANDIDATE_DIR / REQUIRED_FILES[0]).resolve())
        train = load_module("earthformer_multiscale_decoder_train_submission", (CANDIDATE_DIR / REQUIRED_FILES[1]).resolve())
        model_cls = getattr(module, "EarthformerMultiscaleDecoder")
        fusion_cls = getattr(module, "MultiscaleFusionBlock")
        model = train.build_model({"encoder_channels": [8, 16, 32], "hidden_channels": 16, "out_channels": 2})
        features = (torch.randn(2, 8, 3, 8, 8), torch.randn(2, 16, 3, 4, 4), torch.randn(2, 32, 3, 2, 2))
        target = torch.randn(2, 3, 2, 8, 8)
        output = model(features)
        update_check(current_index, output.shape == target.shape and bool(torch.isfinite(output).all()), str(tuple(output.shape)))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["model_structure:top_down_fusion"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn
        module = load_module("earthformer_multiscale_decoder_submission", (CANDIDATE_DIR / REQUIRED_FILES[0]).resolve())
        train = load_module("earthformer_multiscale_decoder_train_submission", (CANDIDATE_DIR / REQUIRED_FILES[1]).resolve())
        model_cls = getattr(module, "EarthformerMultiscaleDecoder")
        fusion_cls = getattr(module, "MultiscaleFusionBlock")
        model = train.build_model({"encoder_channels": [8, 16, 32], "hidden_channels": 16, "out_channels": 2})
        features = (torch.randn(2, 8, 3, 8, 8), torch.randn(2, 16, 3, 4, 4), torch.randn(2, 32, 3, 2, 2))
        target = torch.randn(2, 3, 2, 8, 8)
        output = model(features)
        projected_shapes = [layer(feature).shape for layer, feature in zip(model.lateral_projections, features)]
        module_order = list(model._modules)
        update_check(
            current_index,
            module_order == ["lateral_projections", "fusion_blocks", "output_head"]
            and model.__class__.__name__ == model_cls.__name__
            and len(model.lateral_projections) == 3
            and all(isinstance(layer, nn.Conv3d) for layer in model.lateral_projections)
            and projected_shapes == [(2, 16, 3, 8, 8), (2, 16, 3, 4, 4), (2, 16, 3, 2, 2)]
            and len(model.fusion_blocks) == 2
            and all(block.__class__.__name__ == fusion_cls.__name__ for block in model.fusion_blocks)
            and isinstance(model.output_head, nn.Conv3d),
            f"module_order={module_order}; projected_shapes={[tuple(s) for s in projected_shapes]}",
        )
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["traning:finite_loss_and_update"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn
        module = load_module("earthformer_multiscale_decoder_submission", (CANDIDATE_DIR / REQUIRED_FILES[0]).resolve())
        train = load_module("earthformer_multiscale_decoder_train_submission", (CANDIDATE_DIR / REQUIRED_FILES[1]).resolve())
        model_cls = getattr(module, "EarthformerMultiscaleDecoder")
        fusion_cls = getattr(module, "MultiscaleFusionBlock")
        model = train.build_model({"encoder_channels": [8, 16, 32], "hidden_channels": 16, "out_channels": 2})
        features = (torch.randn(2, 8, 3, 8, 8), torch.randn(2, 16, 3, 4, 4), torch.randn(2, 32, 3, 2, 2))
        target = torch.randn(2, 3, 2, 8, 8)
        output = model(features)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]
        loss = train.train_step(model, features, target, optimizer)
        changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        update_check(
            current_index,
            bool(torch.isfinite(torch.as_tensor(loss))) and changed,
            f"loss={float(loss):.8f} changed={changed}",
        )
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks
