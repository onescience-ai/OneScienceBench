"""V2 adapter for the Earthformer irregular time embedding evaluator."""
from __future__ import annotations

import importlib.util
import json
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

REQUIRED_FILES = ("earthformer_irregular_time_embedding.py", "train.py")
ENV_NAME = "EARTHFORMER_IRREGULAR_TIME_EMBEDDING_SUBMISSION_DIR"
HINTS = ("earthformer_irregular_time_embedding", "earthformer_irregular_time_embedding_V1")
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
    CHECKPOINT_NAMES = ["file:earthformer_irregular_time_embedding.py", "file:train.py"]
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
    CHECKPOINT_NAMES = ["data:irregular_sequence_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        x = torch.randn(2, 4, 3, 4, 5)
        deltas = torch.tensor([[0.0, 1.0, 2.5, 5.0], [0.0, 0.5, 3.0, 7.0]])
        mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
        target = torch.randn(2, 4, 2, 4, 5)
        update_check(
            current_index,
            x.shape == (2, 4, 3, 4, 5)
            and deltas.shape == (2, 4)
            and mask.dtype == torch.bool,
            "x=[B,T,C,H,W], deltas/mask=[B,T]",
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
        train = load_module("earthformer_irregular_time_embedding_train_submission", train_path)
        update_check(
            current_index,
            callable(getattr(train, required_interfaces[0], None))
            and callable(getattr(train, required_interfaces[1], None)),
            "build_model/train_step",
        )
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_equivalence:delta_changes_embedding"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        model_file = config.get("model_file", "earthformer_irregular_time_embedding.py")
        embedding_class = config.get("embedding_class", "IrregularTimeEmbedding")
        model_module = load_module(
            "earthformer_irregular_time_embedding_submission",
            (CANDIDATE_DIR / model_file).resolve(),
        )
        embedding_cls = getattr(model_module, embedding_class)
        embedding = embedding_cls(16, 4)
        emb_a = embedding(torch.tensor([[0.0, 1.0]]))
        emb_b = embedding(torch.tensor([[0.0, 2.0]]))
        update_check(
            current_index,
            not bool(torch.allclose(emb_a, emb_b)),
            "nonuniform delta_t produces distinct embeddings",
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
    CHECKPOINT_NAMES = ["forward_equivalence:delta_changes_forecast"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        model_file = config.get("model_file", "earthformer_irregular_time_embedding.py")
        model_args = config.get(
            "model_args",
            {
                "in_channels": 3,
                "out_channels": 2,
                "hidden_dim": 16,
                "num_heads": 4,
                "num_frequencies": 4,
            },
        )
        train = load_module(
            "earthformer_irregular_time_embedding_train_submission",
            (CANDIDATE_DIR / "train.py").resolve(),
        )
        load_module(
            "earthformer_irregular_time_embedding_submission",
            (CANDIDATE_DIR / model_file).resolve(),
        )
        model = train.build_model(model_args)
        x = torch.randn(2, 4, 3, 4, 5)
        deltas = torch.tensor([[0.0, 1.0, 2.5, 5.0], [0.0, 0.5, 3.0, 7.0]])
        mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
        output = model(x, deltas, mask)
        output_changed = model(x, deltas + 0.25, mask)
        update_check(
            current_index,
            not bool(torch.allclose(output, output_changed)),
            "forecast depends on irregular time deltas",
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
        model_file = config.get("model_file", "earthformer_irregular_time_embedding.py")
        model_args = config.get(
            "model_args",
            {
                "in_channels": 3,
                "out_channels": 2,
                "hidden_dim": 16,
                "num_heads": 4,
                "num_frequencies": 4,
            },
        )
        train = load_module(
            "earthformer_irregular_time_embedding_train_submission",
            (CANDIDATE_DIR / "train.py").resolve(),
        )
        load_module(
            "earthformer_irregular_time_embedding_submission",
            (CANDIDATE_DIR / model_file).resolve(),
        )
        model = train.build_model(model_args)
        x = torch.randn(2, 4, 3, 4, 5)
        deltas = torch.tensor([[0.0, 1.0, 2.5, 5.0], [0.0, 0.5, 3.0, 7.0]])
        mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
        output = model(x, deltas, mask)
        update_check(
            current_index,
            output.shape == (2, 4, 2, 4, 5) and bool(torch.isfinite(output).all()),
            str(tuple(output.shape)),
        )
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["model_structure:temporal_attention_and_time_embedding"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn
        model_file = config.get("model_file", "earthformer_irregular_time_embedding.py")
        model_class = config.get("model_class", "EarthformerIrregularTimeModel")
        embedding_class = config.get("embedding_class", "IrregularTimeEmbedding")
        model_args = config.get(
            "model_args",
            {
                "in_channels": 3,
                "out_channels": 2,
                "hidden_dim": 16,
                "num_heads": 4,
                "num_frequencies": 4,
            },
        )
        module = load_module(
            "earthformer_irregular_time_embedding_submission",
            (CANDIDATE_DIR / model_file).resolve(),
        )
        train = load_module(
            "earthformer_irregular_time_embedding_train_submission",
            (CANDIDATE_DIR / "train.py").resolve(),
        )
        model_cls = getattr(module, model_class)
        embedding_cls = getattr(module, embedding_class)
        model = train.build_model(model_args)
        x = torch.randn(2, 4, 3, 4, 5)
        encoded_shape = model.spatial_encoder(x.reshape(2 * 4, 3, 4, 5)).shape
        module_order = list(model._modules)
        update_check(
            current_index,
            module_order
            == ["spatial_encoder", "time_embedding", "temporal_attention", "norm", "decoder"]
            and model.__class__.__name__ == model_cls.__name__
            and encoded_shape == (8, 16, 4, 5)
            and isinstance(model.spatial_encoder, nn.Conv2d)
            and model.time_embedding.__class__.__name__ == embedding_cls.__name__
            and isinstance(model.temporal_attention, nn.MultiheadAttention)
            and isinstance(model.norm, nn.LayerNorm)
            and isinstance(model.decoder, nn.Conv2d),
            f"module_order={module_order}; encoded_shape={tuple(encoded_shape)}",
        )
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_padding_and_mask(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_executable:masked_steps_zero"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        model_file = config.get("model_file", "earthformer_irregular_time_embedding.py")
        model_args = config.get(
            "model_args",
            {
                "in_channels": 3,
                "out_channels": 2,
                "hidden_dim": 16,
                "num_heads": 4,
                "num_frequencies": 4,
            },
        )
        train = load_module(
            "earthformer_irregular_time_embedding_train_submission",
            (CANDIDATE_DIR / "train.py").resolve(),
        )
        load_module(
            "earthformer_irregular_time_embedding_submission",
            (CANDIDATE_DIR / model_file).resolve(),
        )
        model = train.build_model(model_args)
        x = torch.randn(2, 4, 3, 4, 5)
        deltas = torch.tensor([[0.0, 1.0, 2.5, 5.0], [0.0, 0.5, 3.0, 7.0]])
        mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
        output = model(x, deltas, mask)
        atol = config.get("atol", 1e-7)
        update_check(
            current_index,
            bool(torch.allclose(output[~mask], torch.zeros_like(output[~mask]), atol=atol)),
            "invalid time steps are zeroed",
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
        model_file = config.get("model_file", "earthformer_irregular_time_embedding.py")
        model_args = config.get(
            "model_args",
            {
                "in_channels": 3,
                "out_channels": 2,
                "hidden_dim": 16,
                "num_heads": 4,
                "num_frequencies": 4,
            },
        )
        train = load_module(
            "earthformer_irregular_time_embedding_train_submission",
            (CANDIDATE_DIR / "train.py").resolve(),
        )
        load_module(
            "earthformer_irregular_time_embedding_submission",
            (CANDIDATE_DIR / model_file).resolve(),
        )
        model = train.build_model(model_args)
        x = torch.randn(2, 4, 3, 4, 5)
        deltas = torch.tensor([[0.0, 1.0, 2.5, 5.0], [0.0, 0.5, 3.0, 7.0]])
        mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
        target = torch.randn(2, 4, 2, 4, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3))
        before = [p.detach().clone() for p in model.parameters()]
        loss = train.train_step(model, x, deltas, target, optimizer, mask)
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
