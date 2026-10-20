'''V2 evaluator adapter for the DNABERT-2 inference evaluator.'''
from pathlib import Path
import sys
import ast
import importlib.util
import inspect
import json
import os
import subprocess
import tempfile
import traceback
from typing import Any, Callable, Dict

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

FILENAME, ENV_NAME = "dnabert2_inference.py", "DNABERT2_INFERENCE_SUBMISSION_DIR"
CTOR = {"vocab_size": 7, "hidden_size": 32, "num_heads": 4, "num_layers": 2}
ATTRS = (
    "token_embedding",
    "blocks",
    "blocks.0.attention",
    "blocks.0.attention.qkv",
    "final_norm",
)
EXPECTED = {
    "file": ("file:dnabert2_inference.py", "file:standard:inference_config.json"),
    "checkpoint_loading": (
        "checkpoint_loading:strict_state_dict",
        "checkpoint_loading:uses_torch_load",
        "checkpoint_loading:uses_load_state_dict",
    ),
    "model_structure": (
        "model_structure:nn_module",
        "model_structure:required_components",
        "model_structure:state_dict_contract",
        "model_structure:module_order",
        "model_structure:constructor_contract",
    ),
    "forward_executable": (
        "forward_executable:tensor",
        "forward_executable:shape",
        "forward_executable:finite",
        "forward_executable:deterministic",
        "forward_executable:pooled_shape",
        "forward_executable:pooling_modes",
    ),
    "evaluation_interface": (
        "evaluation_interface:independent_from_onescience",
        "evaluation_interface:cli_flags",
        "evaluation_interface:no_grad",
        "evaluation_interface:cli_smoke",
        "evaluation_interface:artifact",
        "evaluation_interface:reject_partial_checkpoint",
    ),
    "metric": (
        "metric:forward_equivalence",
        "metric:pooling_oracles",
        "metric:artifact_oracle",
        "metric:identifiers_preserved",
    ),
}

_ADAPTER_CONTEXT: dict[str, Any] = {}

def _context(caller_globals, caller_locals):
    if isinstance(caller_locals, dict):
        return caller_locals
    if isinstance(caller_globals, dict):
        return caller_globals
    return _ADAPTER_CONTEXT

def _candidate_file(name: str) -> Path:
    path = (CANDIDATE_DIR / name).resolve()
    path.relative_to(CANDIDATE_DIR)
    return path

def _reference_file(name: str) -> Path:
    path = (REFERENCE_DIR / name).resolve()
    path.relative_to(REFERENCE_DIR)
    return path

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def dotted(value: Any, name: str):
    for part in name.split("."):
        value = getattr(value, part)
    return value

def source_info(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] == "onescience" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "onescience"
        )
        for node in ast.walk(tree)
    )
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return source, imports, names

def fixture(torch):
    tokens = torch.tensor([[6, 1, 2, 3, 4, 5], [6, 4, 3, 2, 1, 0]], dtype=torch.long)
    return tokens, tokens.ne(0)

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["file:dnabert2_inference.py", "file:standard:inference_config.json"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        context = _context(caller_globals, caller_locals)
        required = tuple(config.get("required_files", (FILENAME, "inference_config.json")))
        candidate_name = next((item for item in required if Path(item).name == FILENAME), FILENAME)
        path = _candidate_file(candidate_name)
        reference_path = _reference_file(FILENAME)
        context.update(
            torch=torch,
            path=path,
            candidate=load_module("dnabert2_candidate", path),
            reference=load_module("dnabert2_reference", reference_path),
        )
        config_path = _reference_file("inference_config.json")
        assets = json.loads(config_path.read_text(encoding="utf-8"))["external_assets"]
        ok = (
            assets.get("mount_required") is True
            and Path(assets["weights"]).exists()
            and Path(assets["dataset"]).exists()
        )
        update_check(current_index, True, str(path))
        current_index += 1
        update_check(current_index, ok, str(assets))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "checkpoint_loading:strict_state_dict",
        "checkpoint_loading:uses_torch_load",
        "checkpoint_loading:uses_load_state_dict",
        "evaluation_interface:reject_partial_checkpoint",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        context = _context(caller_globals, caller_locals)
        if "candidate" not in context:
            context["path"] = _candidate_file(config.get("model_file", FILENAME))
            context["candidate"] = load_module("dnabert2_candidate", context["path"])
        if "reference" not in context:
            context["reference"] = load_module("dnabert2_reference", _reference_file(FILENAME))
        context["torch"] = torch
        model = context["candidate"].DNABert2Encoder(**config.get("model_args", CTOR))
        reference = context["reference"].DNABert2Encoder(**config.get("model_args", CTOR))
        strict = config.get("strict", True)
        result = model.load_state_dict(reference.state_dict(), strict=strict)
        manager = context.get("manager")
        if manager is None:
            manager = tempfile.TemporaryDirectory(prefix="dnabert2_infer_")
            context["manager"] = manager
        checkpoint = Path(manager.name) / "reference.pt"
        torch.save({"model_state_dict": reference.state_dict(), "config": CTOR}, checkpoint)
        context.update(
            model=model.eval(),
            reference_model=reference.eval(),
            checkpoint=checkpoint,
        )
        source, _imports, names = source_info(context["path"])
        update_check(
            current_index,
            not result.missing_keys and not result.unexpected_keys,
            str(result),
        )
        current_index += 1
        update_check(current_index, "torch.load" in source, "torch.load")
        current_index += 1
        update_check(current_index, "load_state_dict" in names, str(sorted(names)))
        current_index += 1
        root = Path(manager.name)
        fasta = root / "input.fasta"
        fasta.write_text(">a\nACGTNACG\n>b\nTTGGCCA\n", encoding="utf-8")
        state = reference.state_dict()
        partial = root / "partial.pt"
        torch.save({"model_state_dict": dict(list(state.items())[1:])}, partial)
        invalid_command = [
            sys.executable,
            str(context["path"]),
            "--input-fasta",
            str(fasta),
            "--checkpoint",
            str(partial),
            "--output",
            str(root / "invalid.npz"),
            "--pooling",
            "mean",
            "--device",
            "cpu",
        ]
        rejected = subprocess.run(
            invalid_command,
            capture_output=True,
            text=True,
            timeout=180,
        )
        update_check(
            current_index,
            rejected.returncode != 0,
            f"returncode={rejected.returncode}, stderr={rejected.stderr[-500:]}",
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
    CHECKPOINT_NAMES = [
        "model_structure:nn_module",
        "model_structure:required_components",
        "model_structure:state_dict_contract",
        "model_structure:module_order",
        "model_structure:constructor_contract",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        context = _context(caller_globals, caller_locals)
        if "candidate" not in context:
            context["path"] = _candidate_file(config.get("model_file", FILENAME))
            context["candidate"] = load_module("dnabert2_candidate", context["path"])
        if "reference" not in context:
            context["reference"] = load_module("dnabert2_reference", _reference_file(FILENAME))
        if "model" not in context:
            context["model"] = context["candidate"].DNABert2Encoder(**config.get("model_args", CTOR)).eval()
        if "reference_model" not in context:
            context["reference_model"] = context["reference"].DNABert2Encoder(**config.get("model_args", CTOR)).eval()
        model, reference = context["model"], context["reference_model"]
        missing, mismatched = [], []
        for name in ATTRS:
            try:
                if type(dotted(model, name)).__name__ != type(dotted(reference, name)).__name__:
                    mismatched.append(name)
            except AttributeError:
                missing.append(name)
        state, ref_state = model.state_dict(), reference.state_dict()
        state_ok = list(state) == list(ref_state) and all(
            state[name].shape == ref_state[name].shape for name in state
        )
        children = [(name, type(module).__name__) for name, module in model.named_children()]
        ref_children = [(name, type(module).__name__) for name, module in reference.named_children()]
        signature = inspect.signature(context["candidate"].DNABert2Encoder.__init__)
        update_check(current_index, isinstance(model, torch.nn.Module), type(model).__name__)
        current_index += 1
        update_check(current_index, not missing and not mismatched, f"missing={missing}, mismatched={mismatched}")
        current_index += 1
        update_check(current_index, state_ok, f"keys={len(state)}")
        current_index += 1
        update_check(current_index, children == ref_children, f"candidate={children}, reference={ref_children}")
        current_index += 1
        update_check(current_index, all(name in signature.parameters for name in CTOR), str(signature))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "forward_executable:tensor",
        "forward_executable:shape",
        "forward_executable:finite",
        "forward_executable:deterministic",
        "forward_executable:pooled_shape",
        "forward_executable:pooling_modes",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        context = _context(caller_globals, caller_locals)
        if "candidate" not in context:
            context["path"] = _candidate_file(config.get("model_file", FILENAME))
            context["candidate"] = load_module("dnabert2_candidate", context["path"])
        if "model" not in context:
            context["model"] = context["candidate"].DNABert2Encoder(**config.get("model_args", CTOR)).eval()
        context["torch"] = torch
        model = context["model"]
        tokens, mask = fixture(torch)
        with torch.inference_mode():
            output, repeated = model(tokens, mask), model(tokens, mask)
            pooled, cls = (
                model.embed(tokens, mask, "mean"),
                model.embed(tokens, mask, "cls"),
            )
        update_check(current_index, isinstance(output, torch.Tensor), str(type(output)))
        current_index += 1
        update_check(current_index, tuple(output.shape) == (2, 6, 32), str(tuple(output.shape)))
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output).all()), "finite")
        current_index += 1
        update_check(current_index, torch.equal(output, repeated), "repeat equality")
        current_index += 1
        update_check(current_index, tuple(pooled.shape) == (2, 32), str(tuple(pooled.shape)))
        current_index += 1
        update_check(current_index, not torch.equal(pooled, cls), "mean differs from cls")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "evaluation_interface:independent_from_onescience",
        "evaluation_interface:cli_flags",
        "evaluation_interface:no_grad",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        context = _context(caller_globals, caller_locals)
        files = config.get("files", [FILENAME])
        path = _candidate_file(files[0])
        context["path"] = path
        source, imports, names = source_info(path)
        flags = ("--input-fasta", "--checkpoint", "--output", "--pooling", "--device")
        update_check(current_index, not imports, "no onescience import")
        current_index += 1
        update_check(current_index, all(flag in source for flag in flags), str(flags))
        current_index += 1
        update_check(current_index, bool({"inference_mode", "no_grad"} & names), str(sorted(names)))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks

def check_infer_execution(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["evaluation_interface:cli_smoke", "evaluation_interface:artifact"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        context = _context(caller_globals, caller_locals)
        if "path" not in context:
            context["path"] = _candidate_file(config.get("run_file", FILENAME))
        if "candidate" not in context:
            context["candidate"] = load_module("dnabert2_candidate", context["path"])
        if "reference" not in context:
            context["reference"] = load_module("dnabert2_reference", _reference_file(FILENAME))
        if "reference_model" not in context:
            context["reference_model"] = context["reference"].DNABert2Encoder(**CTOR).eval()
        manager = context.get("manager")
        if manager is None:
            manager = tempfile.TemporaryDirectory(prefix="dnabert2_infer_")
            context["manager"] = manager
        root = Path(manager.name)
        checkpoint = context.get("checkpoint")
        if checkpoint is None:
            checkpoint = root / "reference.pt"
            torch.save({"model_state_dict": context["reference_model"].state_dict(), "config": CTOR}, checkpoint)
            context["checkpoint"] = checkpoint
        fasta, artifact = root / "input.fasta", root / "embeddings.npz"
        fasta.write_text(">a\nACGTNACG\n>b\nTTGGCCA\n", encoding="utf-8")
        mapping = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
        encoded = [
            torch.tensor([6, *(mapping[base] for base in sequence)])
            for sequence in ("ACGTNACG", "TTGGCCA")
        ]
        tokens = torch.zeros(2, max(item.numel() for item in encoded), dtype=torch.long)
        for index, item in enumerate(encoded):
            tokens[index, : item.numel()] = item
        with torch.inference_mode():
            context["expected"] = context["reference_model"].embed(tokens, tokens.ne(0), "mean").numpy()
        command = [
            sys.executable,
            str(context["path"]),
            "--input-fasta",
            str(fasta),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(artifact),
            "--pooling",
            "mean",
            "--device",
            "cpu",
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        context["artifact"] = artifact
        update_check(current_index, process.returncode == 0, process.stderr[-1000:])
        current_index += 1
        update_check(current_index, artifact.is_file(), str(artifact))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["metric:forward_equivalence", "metric:pooling_oracles"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        context = _context(caller_globals, caller_locals)
        if "candidate" not in context:
            context["path"] = _candidate_file(config.get("model_file", FILENAME))
            context["candidate"] = load_module("dnabert2_candidate", context["path"])
        if "reference" not in context:
            context["reference"] = load_module("dnabert2_reference", _reference_file(FILENAME))
        if "model" not in context:
            context["model"] = context["candidate"].DNABert2Encoder(**CTOR).eval()
        if "reference_model" not in context:
            context["reference_model"] = context["reference"].DNABert2Encoder(**CTOR).eval()
        # Each transformed check receives an isolated caller-locals mapping,
        # so the checkpoint-loading check cannot leave its synchronized model
        # available here.  Reapply the legacy evaluator's shared-weight
        # contract before comparing implementation behavior.
        context["model"].load_state_dict(
            context["reference_model"].state_dict(), strict=True
        )
        args = fixture(torch)
        with torch.inference_mode():
            candidate, reference = context["model"](*args), context["reference_model"](*args)
            pooling_matches = {
                mode: torch.allclose(
                    context["model"].embed(*args, mode),
                    context["reference_model"].embed(*args, mode),
                    atol=1e-6,
                    rtol=1e-6,
                )
                for mode in ("mean", "max", "cls")
            }
        forward_ok = torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)
        update_check(current_index, bool(forward_ok), f"max_diff={float((candidate - reference).abs().max())}")
        current_index += 1
        update_check(current_index, all(pooling_matches.values()), str(pooling_matches))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks

def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["metric:artifact_oracle", "metric:identifiers_preserved"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import numpy as np

        context = _context(caller_globals, caller_locals)
        if "artifact" not in context or "expected" not in context:
            candidate_config = config.get("candidate", {})
            run_file = (
                candidate_config.get("run_file", FILENAME)
                if isinstance(candidate_config, dict)
                else FILENAME
            )
            # Transformed checks may receive isolated locals; recreate the
            # inference artifact and oracle target in this check's context.
            check_infer_execution(
                {"run_file": run_file},
                caller_globals=context,
                caller_locals=context,
            )
        with np.load(context["artifact"]) as result:
            actual = result["embeddings"]
            identifiers = result["identifiers"].tolist()
        expected = context["expected"]
        artifact_ok = actual.shape == expected.shape and np.allclose(
            actual, expected, atol=1e-6, rtol=1e-6
        )
        update_check(current_index, bool(artifact_ok), f"actual={actual.shape}, expected={expected.shape}")
        current_index += 1
        update_check(current_index, identifiers == ["a", "b"], str(identifiers))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks
