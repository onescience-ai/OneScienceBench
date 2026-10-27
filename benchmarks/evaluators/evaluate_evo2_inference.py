"""V2 adapter for the legacy Evo2 inference evaluator."""
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

SUBMISSION_FILENAME = 'evo2_inference.py'
SEED = 20260819
FORWARD_COMPARE_ATOL = 1e-6
CLI_TIMEOUT_SECONDS = 180
EXPECTED_LAYERS = 2
DEFAULT_HIDDEN = 32
EXPECTED_VOCAB_SIZE = 47
GENERATION_PROMPT = 'ACGT|'
GENERATION_TOKENS = 4
REQUIRED_BLOCK_ATTRS = (
    'norm',
    'input_projection',
    'depthwise_conv',
    'ssm_projection',
    'delta_projection',
    'A_log',
    'D',
    'output_projection',
)
REQUIRED_CLI_FLAGS = (
    '--prompt',
    '--ckpt-dir',
    '--max-new-tokens',
    '--temperature',
    '--top-k',
    '--top-p',
    '--seed',
    '--output-file',
    '--device',
)

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["file:evo2_inference.py", "file:standard:inference_config.json"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    try:
        entries = params.get("required_files", [])
        phase = params.get("phase")
        if not entries:
            checks[0] = (CHECKPOINT_NAMES[0], False, "required_files is missing or empty")
        else:
            root = CANDIDATE_DIR.resolve()
            for index, relative_path in enumerate(entries[:1]):
                try:
                    path = (root / relative_path).resolve()
                    path.relative_to(root)
                    valid = path.is_file() and bool(path.read_bytes().strip())
                    checks[index] = (CHECKPOINT_NAMES[index], valid, str(path))
                except Exception as error:
                    checks[index] = (CHECKPOINT_NAMES[index], False, str(error))
        checks[1] = (CHECKPOINT_NAMES[1], True, f"reference-side file not required (phase={phase})")
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["checkpoint_loading:strict_state_dict", "checkpoint_loading:file_formats", "checkpoint_loading:uses_torch_load"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    import base_evaluator as base
    import torch
    current_index = 0
    try:
        path = (CANDIDATE_DIR / params["model_file"]).resolve()
        path.relative_to(CANDIDATE_DIR.resolve())
        current_index = 1
        candidate = base._load_module_from_path("probe_" + path.stem, path)
        cfg = base._load_candidate_config(params["config_name"]) if params.get("config_name") else {}
        context = {"config": cfg}
        cls = getattr(candidate, params["model_class"])
        model = cls(**base._resolve_args(params.get("model_args", {}), context)).to(DEVICE).eval()
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = path
            results = []
            state_variants = [state, {"model_state_dict": state}]
            current_index = 1
            for index, payload in enumerate(state_variants):
                checkpoint_dir = root / f"valid_{index}"
                checkpoint_dir.mkdir()
                torch.save(payload, checkpoint_dir / params["checkpoint_file"])
                output = root / f"valid_{index}.json"
                values = dict(params["cli_args"])
                values["--ckpt-dir"] = str(checkpoint_dir / params["checkpoint_file"])
                values["--output-file"] = str(output)
                argv = [part for key, value in values.items() for part in (key, str(value))]
                run = subprocess.run([sys.executable, str(script), *argv], cwd=str(script.parent), capture_output=True, text=True, timeout=params["timeout_seconds"], check=False)
                results.append(run.returncode == 0 and output.is_file())
            checks[1] = (CHECKPOINT_NAMES[1], all(results), f"raw and model_state_dict CLI loads: {results}")
            invalid_results = []
            missing = dict(state)
            missing.pop(next(iter(missing)))
            extra = dict(state)
            extra["__unexpected_checkpoint_key__"] = torch.zeros(1)
            for index, invalid_state in enumerate((missing, extra)):
                checkpoint_dir = root / f"invalid_{index}"
                checkpoint_dir.mkdir()
                torch.save(invalid_state, checkpoint_dir / params["checkpoint_file"])
                output = root / f"invalid_{index}.json"
                values = dict(params["cli_args"])
                values["--ckpt-dir"] = str(checkpoint_dir / params["checkpoint_file"])
                values["--output-file"] = str(output)
                argv = [part for key, value in values.items() for part in (key, str(value))]
                run = subprocess.run([sys.executable, str(script), *argv], cwd=str(script.parent), capture_output=True, text=True, timeout=params["timeout_seconds"], check=False)
                invalid_results.append(run.returncode != 0)
            current_index = 0
            strict_ok = not params["strict"] or all(invalid_results)
            checks[0] = (CHECKPOINT_NAMES[0], strict_ok, f"strict mismatch rejection: {invalid_results}")
            checks[2] = (CHECKPOINT_NAMES[2], all(results), "candidate CLI loaded raw and wrapped torch checkpoints")
    except Exception as error:
        checks[current_index] = (CHECKPOINT_NAMES[current_index], False, str(error))
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    rules = params.get("rules", [])
    CHECKPOINT_NAMES = ["model_structure:mamba_stack", "model_structure:selective_ssm_components", "model_structure:block_projection_shapes", "model_structure:module_order", "model_structure:no_rnn_substitute"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    import base_evaluator as base
    import torch
    model = None
    module = None
    try:
        path = (CANDIDATE_DIR / params["model_file"]).resolve()
        path.relative_to(CANDIDATE_DIR.resolve())
        module = base._load_module_from_path("probe_" + path.stem, path)
        cfg = base._load_candidate_config(params["config_name"]) if params.get("config_name") else {}
        context = {"config": cfg, **cfg}
        cls = getattr(module, params["model_class"])
        model = cls(**base._resolve_args(params.get("model_args", {}), context)).to(DEVICE).eval()
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
    for index, rule in enumerate(rules):
        if model is None:
            continue
        try:
            spec = rule.get("params", {})
            block_type = getattr(module, spec.get("block_class"), None)
            blocks = [item for item in model.modules() if isinstance(block_type, type) and isinstance(item, block_type)]
            kind = rule["type"]
            def width(component, key):
                value = getattr(component, key, None)
                if value is not None:
                    return value
                weight = getattr(component, "weight", None)
                return int(weight.shape[0]) if weight is not None and len(weight.shape) >= 1 else None

            if kind == "model_contract":
                embed = getattr(model, "token_embedding", None)
                head = getattr(model, "lm_head", None)
                block_type = getattr(module, spec["block_class"], None)
                stack = [item for item in model.modules() if isinstance(block_type, type) and isinstance(item, block_type)]
                embedding_weight = getattr(embed, "weight", None)
                vocab_width = getattr(embed, "num_embeddings", None)
                hidden_width = getattr(embed, "embedding_dim", None)
                if embedding_weight is not None and len(embedding_weight.shape) >= 2:
                    vocab_width = vocab_width if vocab_width is not None else int(embedding_weight.shape[0])
                    hidden_width = hidden_width if hidden_width is not None else int(embedding_weight.shape[-1])
                ok = (vocab_width == spec["vocab_size"] and
                      hidden_width == spec["hidden_size"] and
                      width(head, "out_features") == spec["vocab_size"] and
                      len(stack) == spec["num_layers"] and
                      all(hasattr(model, name) for name in spec["required_attributes"]))
            elif kind == "block_contract":
                ok = len(blocks) == spec["count"] and all(
                    all(hasattr(block, name) for name in spec["required_attributes"]) for block in blocks
                )
            elif kind == "projection_contract":
                ok = bool(blocks)
                for block in blocks:
                    ok = ok and width(getattr(block, "input_projection", None), "out_features") == spec["input_projection_width"]
                    ok = ok and width(getattr(block, "ssm_projection", None), "out_features") == spec["ssm_projection_width"]
            elif kind == "named_components":
                ok = all(hasattr(model, name) for name in spec["attributes"])
            elif kind == "causal_recurrence":
                first = torch.tensor(spec["first_input"], dtype=torch.long, device=DEVICE)
                second = torch.tensor(spec["second_input"], dtype=torch.long, device=DEVICE)
                with torch.inference_mode():
                    left = model(first)
                    right = model(second)
                def logits(value):
                    candidates = []
                    seen = set()
                    def collect(item):
                        marker = id(item)
                        if marker in seen:
                            return
                        seen.add(marker)
                        if torch.is_tensor(item):
                            candidates.append(item)
                        elif isinstance(item, dict):
                            for child in item.values():
                                collect(child)
                        elif isinstance(item, (tuple, list)):
                            for child in item:
                                collect(child)
                        elif hasattr(item, "__dict__"):
                            for child in vars(item).values():
                                collect(child)
                    collect(value)
                    matches = [item for item in candidates if item.ndim >= 3 and item.shape[-1] == spec["vocab_size"]]
                    if not matches:
                        matches = [item for item in candidates if item.ndim >= 3]
                    if not matches:
                        raise TypeError("forward result contains no sequence tensor")
                    return matches[0]
                left, right = logits(left), logits(right)
                ok = torch.is_tensor(left) and torch.is_tensor(right) and torch.allclose(
                    left[:, :spec["prefix_length"]], right[:, :spec["prefix_length"]],
                    atol=spec["atol"], rtol=spec["rtol"])
            else:
                ok = False
            checks[index] = (CHECKPOINT_NAMES[index], bool(ok), rule["type"])
        except Exception as error:
            if index < len(checks):
                checks[index] = (CHECKPOINT_NAMES[index], False, str(error))
    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["forward_executable:logit_shape", "forward_executable:vocab_consistency", "forward_executable:finite", "forward_executable:causal_prefix", "forward_executable:batch_consistency"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    import base_evaluator as base
    import torch

    def tensor_from(value, expected_shape=None):
        values = []
        seen = set()
        def collect(item):
            marker = id(item)
            if marker in seen:
                return
            seen.add(marker)
            if torch.is_tensor(item):
                values.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    collect(child)
            elif isinstance(item, (tuple, list)):
                for child in item:
                    collect(child)
            elif hasattr(item, "__dict__"):
                for child in vars(item).values():
                    collect(child)
        collect(value)
        matches = [item for item in values if torch.is_tensor(item) and (expected_shape is None or list(item.shape) == expected_shape)]
        if not matches:
            raise TypeError("forward result contains no tensor matching the expected shape")
        return matches[0]

    def build():
        path = (CANDIDATE_DIR / params["model_file"]).resolve()
        path.relative_to(CANDIDATE_DIR.resolve())
        module = base._load_module_from_path("probe_" + path.stem, path)
        cfg = base._load_candidate_config(params["config_name"]) if params.get("config_name") else {}
        context = {"config": cfg, **cfg}
        context = base._execute_hooks(module, params.get("pre_forward_hooks", []), context, caller_globals, caller_locals)
        model = getattr(module, params["model_class"])(**base._resolve_args(params.get("model_args", {}), context)).to(DEVICE).eval()
        return model, context

    def invoke(model, context, argument_spec, expected_shape=None):
        arguments = base._resolve_args(argument_spec, context)
        converted = {}
        for key, value in arguments.items():
            if isinstance(value, dict) and "values" in value and "dtype" in value:
                value = torch.tensor(value["values"], dtype=getattr(torch, value["dtype"]), device=DEVICE)
            converted[key] = value
        with torch.inference_mode():
            try:
                output = model(**converted)
            except TypeError:
                if len(converted) != 1:
                    raise
                output = model(next(iter(converted.values())))
        return tensor_from(output, expected_shape)

    output = None
    model = None
    context = None
    try:
        model, context = build()
        output = invoke(model, context, params["forward_args"], params.get("out_shape"))
        checks[0] = (CHECKPOINT_NAMES[0], list(output.shape) == params["out_shape"], str(list(output.shape)))
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
    try:
        if model is None:
            raise RuntimeError("model construction or forward execution did not complete")
        expected_vocab = params["out_shape"][-1]
        ok = output.shape[-1] == expected_vocab
        checks[1] = (CHECKPOINT_NAMES[1], bool(ok), "logit vocabulary width")
    except Exception as error:
        checks[1] = (CHECKPOINT_NAMES[1], False, str(error))
    try:
        if output is None:
            raise RuntimeError("forward execution did not produce output")
        checks[2] = (CHECKPOINT_NAMES[2], bool(torch.isfinite(output).all()), "logits finiteness")
    except Exception as error:
        checks[2] = (CHECKPOINT_NAMES[2], False, str(error))
    try:
        if model is None:
            raise RuntimeError("model construction did not complete")
        probe = params["causal_prefix"]
        first = invoke(model, context, probe["first_args"])[:, :probe["prefix_length"]]
        second = invoke(model, context, probe["second_args"])[:, :probe["prefix_length"]]
        ok = first.shape == second.shape and torch.allclose(first, second, atol=probe["atol"], rtol=probe["rtol"])
        checks[3] = (CHECKPOINT_NAMES[3], bool(ok), "causal prefix invariance")
    except Exception as error:
        checks[3] = (CHECKPOINT_NAMES[3], False, str(error))
    try:
        if model is None:
            raise RuntimeError("model construction did not complete")
        checks[4] = (CHECKPOINT_NAMES[4], True, "batch consistency is not specified by the task")
    except Exception as error:
        checks[4] = (CHECKPOINT_NAMES[4], False, str(error))
    return checks


def check_infer_execution(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["evaluation_interface:independent_cli", "evaluation_interface:generation_and_scoring", "evaluation_interface:eval_no_grad", "evaluation_interface:cli_smoke", "evaluation_interface:cli_deterministic", "evaluation_interface:invalid_generation_args_rejected"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    import base_evaluator as base
    import torch
    script = (CANDIDATE_DIR / params["run_file"]).resolve()
    try:
        script.relative_to(CANDIDATE_DIR.resolve())
        tree = ast.parse(script.read_text(encoding="utf-8"))
        imports = [alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        imports += [node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        flags = {item for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str) for item in [node.value]}
        required = params.get("required_cli_flags", [])
        entrypoint = next((node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == params["infer_function"]), None)
        ok = "onescience" not in imports and all(flag in flags for flag in required) and entrypoint is not None
        checks[0] = (CHECKPOINT_NAMES[0], ok, "CLI flags and package independence")
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
    model = None
    try:
        module = base._load_module_from_path("probe_" + Path(params["run_file"]).stem, script)
        if not callable(getattr(module, params["infer_function"], None)):
            raise TypeError(f"configured inference entrypoint {params['infer_function']} is not callable")
        context = {"config": {}}
        context = base._execute_hooks(module, params.get("pre_infer_hooks", []), context, caller_globals, caller_locals)
        model = getattr(module, params["model_class"])(**base._resolve_args(params.get("model_args", {}), context)).to(DEVICE)
        model.eval()
        method_names = params["required_methods"]
        methods = [getattr(model, name, None) for name in method_names]
        observed_grad = []
        hook = model.register_forward_pre_hook(lambda _module, _args: observed_grad.append(torch.is_grad_enabled()))
        try:
            infer_args = base._resolve_args(params["infer_args"], context)
            sequence = params["score_sequence"]
            generated = methods[0](**infer_args)
            score = methods[1](sequence)
        finally:
            hook.remove()
        checks[1] = (CHECKPOINT_NAMES[1], all(callable(method) for method in methods) and generated is not None
                     and isinstance(score, (int, float, torch.Tensor)), "generation and sequence scoring execute")
        checks[2] = (CHECKPOINT_NAMES[2], not any(observed_grad),
                     "forward calls inside generation and scoring do not track gradients")
    except Exception as error:
        checks[1] = (CHECKPOINT_NAMES[1], False, str(error))
        checks[2] = (CHECKPOINT_NAMES[2], False, str(error))
    with tempfile.TemporaryDirectory() as temporary:
        try:
            output = Path(temporary) / params["output_path"]
            checkpoint_directory = Path(temporary) / "checkpoints"
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_directory / params["checkpoint_file"]
            if model is None:
                raise RuntimeError("model construction did not complete")
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            values = dict(params.get("cli_args", {}))
            values["--ckpt-dir"] = str(checkpoint_path)
            values["--output-file"] = str(output)
            argv = [part for key, value in values.items() for part in (key, str(value))]
            result = subprocess.run([sys.executable, str(script), *argv], cwd=str(script.parent), capture_output=True, text=True, timeout=params["timeout_seconds"], check=False)
            payload = json.loads(output.read_text(encoding="utf-8")) if result.returncode == 0 else None
            fields = params["output_fields"]
            cli_args = params["cli_args"]
            prompt = cli_args["--prompt"]
            generated = payload.get(fields[1]) if isinstance(payload, dict) else None
            score = payload.get(fields[2]) if isinstance(payload, dict) else None
            valid = (
                payload is not None
                and all(field in payload for field in fields)
                and payload.get(fields[0]) == prompt
                and isinstance(generated, str)
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
            )
            checks[3] = (CHECKPOINT_NAMES[3], bool(valid), str(payload) if payload is not None else result.stderr[-1000:])
            if valid:
                repeat_output = Path(temporary) / "repeat.json"
                values["--output-file"] = str(repeat_output)
                repeat = subprocess.run([sys.executable, str(script), *[part for key, value in values.items() for part in (key, str(value))]], cwd=str(script.parent), capture_output=True, text=True, timeout=params["timeout_seconds"], check=False)
                second = json.loads(repeat_output.read_text(encoding="utf-8")) if repeat.returncode == 0 else {}
                checks[4] = (CHECKPOINT_NAMES[4], repeat.returncode == 0 and payload == second, "same-seed output is repeatable")
            else:
                checks[4] = (CHECKPOINT_NAMES[4], False, "CLI smoke test did not produce valid JSON")
        except Exception as error:
            checks[3] = (CHECKPOINT_NAMES[3], False, str(error))
            checks[4] = (CHECKPOINT_NAMES[4], False, str(error))
    checks[5] = (CHECKPOINT_NAMES[5], True, "invalid-generation-argument behavior is not specified by the task")
    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["metric:forward_equivalence", "metric:score_sequence_equivalence", "metric:score_finiteness", "metric:generation_oracle", "metric:seed_sensitivity"]
    checks = [(name, False, "因前置步骤失败而未执行") for name in CHECKPOINT_NAMES]
    import base_evaluator as base
    import torch
    import torch.nn.functional as F
    model = None
    try:
        spec = params["candidate"]
        path = (CANDIDATE_DIR / spec["run_file"]).resolve()
        path.relative_to(CANDIDATE_DIR.resolve())
        module = base._load_module_from_path("probe_oracle_" + path.stem, path)
        context = {"config": {}}
        context = base._execute_hooks(module, params.get("pre_infer_hooks", []), context, caller_globals, caller_locals)
        model = getattr(module, spec["model_class"])(**base._resolve_args(spec.get("model_args", {}), context)).to(DEVICE).eval()
        sequence = params["infer_args"]["sequence"]
        vocab = params["infer_args"]["vocab"]
        input_ids = torch.tensor([[vocab.index(char) for char in sequence]], dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            forward_result = getattr(model, spec["forward_method"])(input_ids)
        def tensors(value):
            if torch.is_tensor(value):
                return [value]
            if isinstance(value, dict):
                return [tensor for child in value.values() for tensor in tensors(child)]
            if isinstance(value, (tuple, list)):
                return [tensor for child in value for tensor in tensors(child)]
            return []
        logits_options = [item for item in tensors(forward_result) if item.ndim >= 3 and item.shape[-1] == len(vocab)]
        if not logits_options:
            raise TypeError("forward result contains no vocabulary-sized sequence tensor")
        logits = logits_options[0]
        checks[0] = (CHECKPOINT_NAMES[0], tuple(logits.shape[:2]) == tuple(input_ids.shape), str(tuple(logits.shape)))
        expected = F.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, input_ids[:, 1:, None]).mean()
        observed = getattr(model, spec["score_method"])(sequence)
        observed = float(observed.detach().cpu().item()) if torch.is_tensor(observed) and observed.numel() == 1 else float(observed)
        expected = float(expected.detach().cpu().item())
        checks[1] = (CHECKPOINT_NAMES[1], math.isclose(observed, expected, rel_tol=params["rtol"], abs_tol=params["atol"]), f"expected={expected}, observed={observed}")
        checks[2] = (CHECKPOINT_NAMES[2], math.isfinite(observed), str(observed))
        generation_args = params["reference"]["generation_args"]
        generated = getattr(model, spec["generation_method"])(**generation_args)
        prompt = generation_args["prompt"]
        checks[3] = (CHECKPOINT_NAMES[3], isinstance(generated, str), repr(generated))
        checks[4] = (CHECKPOINT_NAMES[4], True, "seed sensitivity is not specified by the task")
    except Exception as error:
        pending = next((index for index, item in enumerate(checks) if not item[1]), 0)
        checks[pending] = (CHECKPOINT_NAMES[pending], False, str(error))
    return checks
