"""V2 evaluator adapter for the Evo2 training evaluator."""
from pathlib import Path
import sys
import ast
import importlib.util
import inspect
import json
import math
import os
import subprocess
import tempfile
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

FILENAME = "evo2_train.py"
STANDARD_CONFIG = "train_config.json"
SEED = 20260819
FORWARD_COMPARE_ATOL = 1e-6
DEFAULT_HIDDEN = 32
DEFAULT_VOCAB = 5
EXPECTED_LAYERS = 2
REQUIRED_BLOCK_ATTRS = (
    "norm", "input_projection", "depthwise_conv", "ssm_projection",
    "delta_projection", "A_log", "D", "output_projection",
)
_STATE = {}

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _candidate():
    module = _STATE.get("candidate")
    if module is None:
        module = _load("evo2_train_submission", (CANDIDATE_DIR / FILENAME).resolve())
        _STATE["candidate"] = module
    return module

def _reference():
    module = _STATE.get("reference")
    if module is None:
        module = _load("evo2_train_reference", (REFERENCE_DIR / FILENAME).resolve())
        _STATE["reference"] = module
    return module

def _run_cli(script: Path, arguments: list[str]):
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=str(script.parent), capture_output=True, text=True,
        timeout=180, check=False,
    )

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:evo2_train.py", "file:standard:train_config.json"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        torch.manual_seed(SEED)
        candidate_path = (CANDIDATE_DIR / FILENAME).resolve()
        reference_path = (REFERENCE_DIR / FILENAME).resolve()
        candidate = _load("evo2_train_submission", candidate_path)
        reference = _load("evo2_train_reference", reference_path)
        _STATE.update(candidate=candidate, reference=reference, submission_path=candidate_path)
        ok, msg = candidate_path.is_file(), str(candidate_path)
        update_check(current_index, ok, msg)
        current_index += 1
        standard_path = (REFERENCE_DIR / STANDARD_CONFIG).resolve()
        ok = standard_path.is_file()
        detail = str(standard_path)
        if ok:
            payload = json.loads(standard_path.read_text(encoding="utf-8"))
            ok = isinstance(payload, dict)
            if ok:
                assets = payload.get("external_assets")
                if not isinstance(assets, dict) or assets.get("mount_required") is not True:
                    ok, detail = False, "train_config.json must declare mount_required=true external_assets"
                else:
                    missing = [
                        name for name in ("weights", "dataset")
                        if not isinstance(assets.get(name), str)
                        or not assets[name].startswith("/public/share/")
                    ]
                    if missing:
                        ok, detail = False, f"external assets must use onestore paths for: {missing}"
                    else:
                        detail = "complete weights and dataset are declared as externally mounted assets"
            else:
                detail = f"{standard_path} (JSON object required)"
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "data:sequence_encoding", "data:case_normalization",
        "data:invalid_sequence_rejected", "data:empty_sequence_rejected",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        candidate = _candidate()
        encode = candidate.encode_sequence
        encoded = encode("ACGTN")
        ok = (
            isinstance(encoded, torch.Tensor)
            and encoded.dtype == torch.long
            and tuple(encoded.shape) == (5,)
            and encoded.tolist() == [0, 1, 2, 3, 4]
        )
        update_check(current_index, ok, f"ids={encoded.tolist() if isinstance(encoded, torch.Tensor) else encoded!r}")
        current_index += 1
        lowered = encode("acgtn")
        ok = isinstance(lowered, torch.Tensor) and lowered.tolist() == [0, 1, 2, 3, 4]
        update_check(current_index, ok, "lowercase input is upper-cased before tokenization")
        current_index += 1
        invalid_ok, invalid_detail = False, "invalid base accepted"
        try:
            encode("ACGX")
        except ValueError:
            invalid_ok, invalid_detail = True, "invalid base rejected with ValueError"
        except Exception as error:
            invalid_detail = f"{type(error).__name__}: {error}"
        update_check(current_index, invalid_ok, invalid_detail)
        current_index += 1
        empty_ok, empty_detail = False, "empty sequence accepted"
        try:
            encode("")
        except ValueError:
            empty_ok, empty_detail = True, "empty sequence rejected with ValueError"
        except Exception as error:
            empty_detail = f"{type(error).__name__}: {error}"
        update_check(current_index, empty_ok, empty_detail)
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "train_script:independent_training", "train_script:grad_clipping",
        "train_script:mock_data_cli", "train_script:mamba_contract",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        path = (CANDIDATE_DIR / FILENAME).resolve()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        ones = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "onescience" for alias in node.names):
                ones = True
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "onescience":
                ones = True
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        independent = not ones and "AdamW" in attrs and "backward" in attrs
        update_check(current_index, independent, "no onescience import; AdamW optimizer and backward pass are present")
        current_index += 1
        clipped = "clip_grad_norm_" in attrs or "clip_grad_norm" in attrs
        update_check(current_index, clipped, "gradient clipping is applied during training")
        current_index += 1
        flags = ("--dataset-config", "--mock-data", "--output-dir", "--max-steps", "--seed")
        update_check(current_index, all(flag in source for flag in flags), f"required CLI flags present: {', '.join(flags)}")
        current_index += 1
        tokens = ("Evo2MambaBlock", "A_log", "depthwise_conv", "ssm_projection")
        update_check(current_index, all(token in source for token in tokens), "selective state-space block components are declared in the source")
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "model_structure:mamba_stack", "model_structure:selective_ssm_components",
        "model_structure:block_projection_shapes", "model_structure:module_order",
        "model_structure:no_rnn_substitute",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        from torch import nn
        candidate = _candidate()
        model = candidate.Evo2TrainModel()
        _STATE["model"] = model
        layer_cls = getattr(candidate, "Evo2MambaBlock", None)
        stack_ok = (
            isinstance(model.token_embedding, nn.Embedding)
            and model.token_embedding.num_embeddings == DEFAULT_VOCAB
            and model.token_embedding.embedding_dim == DEFAULT_HIDDEN
            and isinstance(model.mamba_layers, nn.ModuleList)
            and len(model.mamba_layers) == EXPECTED_LAYERS
            and inspect.isclass(layer_cls)
            and all(isinstance(layer, layer_cls) for layer in model.mamba_layers)
            and isinstance(model.final_norm, nn.LayerNorm)
            and model.final_norm.normalized_shape == (DEFAULT_HIDDEN,)
            and isinstance(model.lm_head, nn.Linear)
            and model.lm_head.in_features == DEFAULT_HIDDEN
            and model.lm_head.out_features == DEFAULT_VOCAB
        )
        update_check(current_index, stack_ok, str(model))
        current_index += 1
        layer = model.mamba_layers[0]
        components_ok = (
            all(hasattr(layer, name) for name in REQUIRED_BLOCK_ATTRS)
            and isinstance(layer.depthwise_conv, nn.Conv1d)
            and layer.depthwise_conv.groups == layer.depthwise_conv.in_channels == DEFAULT_HIDDEN
        )
        update_check(current_index, components_ok, f"required={REQUIRED_BLOCK_ATTRS}; depthwise conv groups={getattr(layer.depthwise_conv, 'groups', None)}")
        current_index += 1
        probes_ok, probe_details = True, []
        with torch.no_grad():
            for name, raw_input, expected_out in (
                ("input_projection", torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN * 2)),
                ("depthwise_conv", torch.randn(2, DEFAULT_HIDDEN, 7), (2, DEFAULT_HIDDEN, 4)),
                ("ssm_projection", torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN * 3)),
                ("delta_projection", torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN)),
                ("output_projection", torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN)),
                ("norm", torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN)),
            ):
                module = getattr(layer, name, None)
                try:
                    output = module(raw_input)
                    shape_ok = isinstance(output, torch.Tensor) and tuple(output.shape) == expected_out
                except Exception as error:
                    shape_ok, output = False, None
                    probe_details.append(f"{name}:{type(error).__name__}")
                if not shape_ok:
                    probe_details.append(f"{name}->{tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__}!={expected_out}")
                    probes_ok = False
        a_log_ok = hasattr(layer, "A_log") and tuple(layer.A_log.shape) == (DEFAULT_HIDDEN,)
        d_ok = hasattr(layer, "D") and tuple(layer.D.shape) == (DEFAULT_HIDDEN,)
        if not (a_log_ok and d_ok):
            probes_ok = False
            probe_details.append(f"A_log={a_log_ok} D={d_ok}")
        update_check(current_index, probes_ok, "; ".join(probe_details) if probe_details else "gated input, 3-way ssm projection, delta and output projections match the contract")
        current_index += 1
        expected_order = ["token_embedding", "mamba_layers", "final_norm", "lm_head"]
        actual_order = [name for name, _child in model.named_children()]
        update_check(current_index, actual_order[:len(expected_order)] == expected_order, f"expected prefix={expected_order}, actual={actual_order}")
        current_index += 1
        rnn_ok = not any(isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)) for module in model.modules())
        update_check(current_index, rnn_ok, "the causal stack must use selective state-space blocks, not GRU/LSTM/RNN")
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "forward_executable:logit_shape", "forward_executable:finite",
        "forward_executable:causal_prefix", "forward_executable:batch_consistency",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        model = _STATE.get("model")
        if model is None:
            model = _candidate().Evo2TrainModel()
            _STATE["model"] = model
        model = model.eval()
        tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        with torch.inference_mode():
            output = model(tokens)
        _STATE.update(tokens=tokens, output=output)
        ok = isinstance(output, torch.Tensor) and tuple(output.shape) == (1, 4, DEFAULT_VOCAB)
        update_check(current_index, ok, f"shape={tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__}")
        current_index += 1
        ok = isinstance(output, torch.Tensor) and bool(torch.isfinite(output).all())
        update_check(current_index, ok, "all logits finite")
        current_index += 1
        with torch.inference_mode():
            prefix_a = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))[:, :3]
            prefix_b = model(torch.tensor([[0, 1, 2, 4]], dtype=torch.long))[:, :3]
        ok = torch.allclose(prefix_a, prefix_b, atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        update_check(current_index, ok, "changing a future token leaves prefix logits unchanged")
        current_index += 1
        batch = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
        with torch.inference_mode():
            batched = model(batch)
        ok = tuple(batched.shape) == (2, 4, DEFAULT_VOCAB) and torch.allclose(
            batched[0], batched[1], atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL
        )
        update_check(current_index, ok, "identical sequences in one batch produce identical logits")
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_equivalence:logits"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        reference_model = _reference().Evo2TrainModel().eval()
        candidate_model = _STATE.get("model")
        if candidate_model is None:
            candidate_model = _candidate().Evo2TrainModel()
        candidate_model = candidate_model.eval()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        _STATE["model"] = candidate_model
        torch.manual_seed(SEED)
        batches = [
            torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=torch.long),
        ]
        equivalent, max_diff = True, 0.0
        with torch.inference_mode():
            for batch in batches:
                expected = reference_model(batch)
                generated = candidate_model(batch)
                if tuple(generated.shape) != tuple(expected.shape):
                    equivalent = False
                    break
                max_diff = max(max_diff, float((expected - generated).abs().max()))
        equivalent = equivalent and max_diff <= FORWARD_COMPARE_ATOL
        update_check(current_index, equivalent, f"fixed-weight logits match the standard implementation; max_abs_diff={max_diff:.3e}, atol={FORWARD_COMPARE_ATOL}")
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "traning:short_fit", "traning:parameter_update",
        "traning:cli_mock_fit", "traning:cli_config_fit",
        "traning:cli_exclusive_args",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        candidate = _candidate()
        manager = tempfile.TemporaryDirectory()
        output_dir = Path(manager.name)
        _STATE.update(temp_manager=manager, output_dir=output_dir)
        model = candidate.Evo2TrainModel()
        parameter_before = next(model.parameters()).detach().clone()
        trainer = candidate.Evo2Trainer(model, output_dir, learning_rate=1e-3, seed=11)
        history = trainer.fit(["ACGTACGT", "TGCATGCA"], max_steps=2)
        _STATE["history"] = history
        finite = (
            isinstance(history, list) and len(history) == 2
            and all(isinstance(row, dict) and math.isfinite(float(row.get("train_loss", float("nan")))) for row in history)
        )
        update_check(current_index, finite, f"two-step mock fit produced {len(history) if isinstance(history, list) else 0} metric rows")
        current_index += 1
        updated = not torch.equal(parameter_before, next(model.parameters()).detach())
        update_check(current_index, updated, "optimizer steps changed the model parameters")
        current_index += 1
        cli = (CANDIDATE_DIR / FILENAME).resolve()
        mock_output = output_dir / "cli_mock"
        mock_run = _run_cli(cli, ["--mock-data", "--output-dir", str(mock_output), "--max-steps", "2", "--seed", "11"])
        ok = mock_run.returncode == 0 and (mock_output / "metrics.jsonl").is_file()
        update_check(current_index, ok, f"returncode={mock_run.returncode}; stderr={mock_run.stderr.strip()[:200]}")
        current_index += 1
        config_path = output_dir / "dataset_config.json"
        config_path.write_text(json.dumps({"sequences": ["ACGTACGT", "TGCATGCA"]}), encoding="utf-8")
        config_output = output_dir / "cli_config"
        config_run = _run_cli(cli, ["--dataset-config", str(config_path), "--output-dir", str(config_output), "--max-steps", "2", "--seed", "11"])
        ok = config_run.returncode == 0 and (config_output / "evo2_train.pt").is_file()
        update_check(current_index, ok, f"returncode={config_run.returncode}; stderr={config_run.stderr.strip()[:200]}")
        current_index += 1
        both_run = _run_cli(cli, ["--mock-data", "--dataset-config", str(config_path), "--output-dir", str(output_dir / "cli_both"), "--max-steps", "2"])
        neither_run = _run_cli(cli, ["--output-dir", str(output_dir / "cli_neither"), "--max-steps", "2"])
        ok = both_run.returncode != 0 and neither_run.returncode != 0
        update_check(current_index, ok, f"both={both_run.returncode}, neither={neither_run.returncode}; exactly one of --dataset-config/--mock-data must be given")
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifacts:checkpoint_and_metrics", "artifacts:metrics_rows"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        output_dir = _STATE.get("output_dir")
        if output_dir is None:
            raise RuntimeError("training stage did not run; artifacts cannot be inspected")
        metrics_path = output_dir / "metrics.jsonl"
        checkpoint_path = output_dir / "evo2_train.pt"
        ok = metrics_path.is_file() and checkpoint_path.is_file()
        detail = f"metrics.jsonl={metrics_path.is_file()}, evo2_train.pt={checkpoint_path.is_file()}"
        if ok:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            required = {"format", "model_state_dict", "optimizer_state_dict", "completed_steps"}
            ok = isinstance(checkpoint, dict) and required <= set(checkpoint)
            detail = f"checkpoint fields={sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__}"
        update_check(current_index, ok, detail)
        current_index += 1
        rows_ok, rows_detail = False, "metrics.jsonl missing"
        if metrics_path.is_file():
            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            rows_ok = len(rows) == 2 and all(
                {"step", "train_loss"} <= set(row) and math.isfinite(float(row["train_loss"]))
                for row in rows
            )
            rows_detail = f"{len(rows)} rows with finite train_loss"
        update_check(current_index, rows_ok, rows_detail)
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks
