"""V2 evaluator adapter for MeshGraphNet CylinderFlow training."""
import ast
from collections import Counter
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Mapping, Sequence

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

REQUIRED_FILES = ("model.py", "dataset.py", "utils.py", "config.yaml", "train.py")
MODEL_DEFAULTS = {"node_input_dim": 6, "edge_input_dim": 3, "output_dim": 3, "hidden_dim": 128, "mlp_hidden_layers": 2, "processor_steps": 15, "aggregation": "sum"}
TRAIN_DEFAULTS = {"epochs": 25, "batch_size": 1, "learning_rate": 1.0e-4, "lr_decay_rate": 0.9999991}
SEED = 20260821
NODE_COUNT = 8
EDGE_INDEX = ((0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4), (1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 7, 6, 4, 7))
FORWARD_ATOL = 1.0e-5
FORWARD_RTOL = 1.0e-5

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _project_source(directory: Path, names: Sequence[str]) -> str:
    return "\n".join(_read(directory / name) for name in names if (directory / name).is_file())

def _compact(source: str) -> str:
    return "".join(source.lower().split())

def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(_read(path))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value

def _section(config: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = config.get(name)
        if isinstance(value, Mapping):
            return value
    return {}

def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    return next((mapping[name] for name in names if name in mapping), None)

def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= max(abs(expected), 1.0) * 1.0e-9
        except (TypeError, ValueError):
            return False
    return actual == expected

def _load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    directory = str(path.parent)
    while directory in sys.path:
        sys.path.remove(directory)
    sys.path.insert(0, directory)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _build_model(module: Any) -> Any:
    model_class = getattr(module, "MeshGraphNet", None)
    if not inspect.isclass(model_class):
        raise TypeError("model.py must define class MeshGraphNet")
    signature = inspect.signature(model_class)
    aliases = {"node_input_dim": 6, "node_dim": 6, "edge_input_dim": 3, "edge_dim": 3, "output_dim": 3, "out_dim": 3, "hidden_dim": 128, "latent_dim": 128, "processor_steps": 15, "message_passing_steps": 15, "num_message_passing_steps": 15, "mlp_hidden_layers": 2, "hidden_layers": 2, "aggregation": "sum", "activation": "relu"}
    kwargs = {name: value for name, value in aliases.items() if name in signature.parameters}
    return model_class(**kwargs)

def _tiny_graph(torch: Any) -> tuple[Any, Any, Any, Any]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    node = torch.randn((NODE_COUNT, 6), generator=generator)
    edge_index = torch.tensor(EDGE_INDEX, dtype=torch.long)
    edge = torch.randn((edge_index.shape[1], 3), generator=generator)
    target = torch.randn((NODE_COUNT, 3), generator=generator)
    return node, edge, edge_index, target

def _module_list(model: Any) -> list[Any]:
    for name in ("processor", "processors", "message_passing_blocks", "processor_blocks"):
        value = getattr(model, name, None)
        if value is not None:
            try:
                return list(value)
            except TypeError:
                pass
    return []

def _parameter_shapes(model: Any) -> Counter[tuple[int, ...]]:
    return Counter(tuple(parameter.shape) for parameter in model.parameters())

def _canonicalize_model(torch: Any, model: Any) -> None:
    import torch.nn as nn
    with torch.no_grad():
        for index, module in enumerate(model.modules()):
            if isinstance(module, nn.Linear):
                values = torch.arange(module.weight.numel(), dtype=torch.float64)
                values = ((values.remainder(29) - 14.0) / 5000.0).reshape(module.weight.shape)
                module.weight.copy_(values.to(module.weight.dtype))
                if module.bias is not None:
                    module.bias.fill_((index % 7 - 3) / 1000.0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    module.weight.fill_(1.0)
                if module.bias is not None:
                    module.bias.zero_()

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check required files and parseability."""
    CHECKPOINT_NAMES = ["required:model.py", "parse:model.py", "required:dataset.py", "parse:dataset.py", "required:utils.py", "parse:utils.py", "required:config.yaml", "parse:config.yaml", "required:train.py", "parse:train.py"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        for name in REQUIRED_FILES:
            path = CANDIDATE_DIR / name
            present = path.is_file()
            update_check(current_index, present, path)
            current_index += 1
            if not present:
                current_index += 1
                continue
            try:
                if path.suffix == ".py":
                    ast.parse(_read(path), filename=str(path))
                else:
                    _load_yaml(path)
                update_check(current_index, True, "valid")
            except Exception as exc:
                update_check(current_index, False, f"{type(exc).__name__}: {exc}")
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_datasset_pipeline(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check TFRecord schema, graph construction, targets, noise, and statistics."""
    CHECKPOINT_NAMES = ["data:tfrecord_fields", "data:trajectory_limits", "data:bidirectional_unique_edges", "data:edge_features", "data:node_type_one_hot", "data:teacher_forcing_target", "data:normal_only_noise", "data:train_only_statistics", "data:one_step_length", "data:disjoint_graph_collate"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        path = CANDIDATE_DIR / "dataset.py"
        if not path.is_file():
            update_check(current_index, False, path)
            current_index += 1
            current_index = len(CHECKPOINT_NAMES)
        else:
            source = _read(path)
            lower = source.lower()
            compact = _compact(source)
            config_data = _load_yaml(CANDIDATE_DIR / "config.yaml")
            data = _section(config_data, "data", "dataset")
            train_count = _first(data, ("train_trajectories", "train_samples"))
            valid_count = _first(data, ("valid_trajectories", "val_trajectories", "val_samples"))
            test_count = _first(data, ("test_trajectories", "test_samples"))
            steps = _first(data, ("steps", "train_steps"))
            noise = _first(data, ("noise_std", "noise"))
            velocity_next = "velocity" in lower and any(token in compact for token in ("velocity[time_index+1]", "velocity[t+1]", "velocity[idx+1]"))
            velocity_delta = velocity_next and any(token in compact for token in ("-trajectory.velocity[", "-velocity[", "torch.sub("))
            pressure_next = "pressure" in lower and any(token in compact for token in ("pressure[time_index+1]", "pressure[t+1]", "pressure[idx+1]"))
            ok = "tfrecorddataset" in compact and "decode_raw" in lower and all(name in lower for name in ("cells", "mesh_pos", "node_type", "velocity", "pressure")); update_check(current_index, ok, "official CylinderFlow TFRecord fields"); current_index += 1
            ok = _same(train_count, 400) and _same(valid_count, 10) and _same(test_count, 10) and _same(steps, 300); update_check(current_index, ok, f"train/valid/test={train_count}/{valid_count}/{test_count}, steps={steps}"); current_index += 1
            ok = "unique" in lower and any(token in compact for token in ("[:,::-1]", "[::-1]", ".flip(", "senders,receivers")) and all(token in compact for token in ("[0,1]", "[1,2]", "[2,0]")); update_check(current_index, ok, "triangle edges are deduplicated and made bidirectional"); current_index += 1
            ok = ("displacement" in lower or "relative" in lower) and any(token in lower for token in ("vector_norm", "linalg.norm", "torch.norm")) and "cat" in lower; update_check(current_index, ok, "[dx,dy,L2 distance]"); current_index += 1
            ok = "one_hot" in lower and (all(token in source for token in ("NORMAL", "INFLOW", "OUTFLOW", "WALL_BOUNDARY")) or ("squeezed-3" in compact and "normal" in lower)) and "num_classes=4" in compact; update_check(current_index, ok, "NORMAL/INFLOW/OUTFLOW/WALL_BOUNDARY -> four channels"); current_index += 1
            update_check(current_index, velocity_delta and pressure_next, "velocity increment and next-step pressure"); current_index += 1
            ok = _same(noise, 0.02) and "normal_mask" in lower and "noise" in lower and any(token in compact for token in ("-=noise", "-noise", "torch.sub(")); update_check(current_index, ok, f"NORMAL-node train noise std={noise}"); current_index += 1
            ok = "edge_stats.json" in lower and "node_stats.json" in lower and "training" in lower and all(token in lower for token in ("velocity_diff_mean", "pressure_std")); update_check(current_index, ok, "supplied stats preferred; fallback derives from training trajectories"); current_index += 1
            ok = "steps-1" in compact and "divmod" in lower; update_check(current_index, ok, "each trajectory contributes steps-1 adjacent-frame samples"); current_index += 1
            ok = "edge_index+offset" in compact and "torch.cat" in lower; update_check(current_index, ok, "node offsets are applied when batching graphs"); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check optimizer, scheduler, DDP/AMP, checkpoint, and rollout contracts."""
    CHECKPOINT_NAMES = ["train:cli", "train:model_defaults", "train:training_defaults", "train:adam_step_scheduler", "train:normalized_mse", "train:optimization_loop", "train:ddp_amp", "train:resume_early_stopping", "train:best_last_checkpoints", "train:rollout_horizons", "train:rollout_boundary_rule", "train:rollout_metrics", "train:result_files", "train:default_output"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        config_data = _load_yaml(CANDIDATE_DIR / "config.yaml")
        training = _section(config_data, "training", "train")
        evaluation = _section(config_data, "evaluation", "eval")
        model = _section(config_data, "model")
        train_source = _read(CANDIDATE_DIR / "train.py")
        source = _project_source(CANDIDATE_DIR, ("train.py", "utils.py", "dataset.py"))
        lower = source.lower(); compact = _compact(source)
        model_mismatches = [f"{name}={model.get(name)!r}" for name, expected in MODEL_DEFAULTS.items() if name in model and not _same(model.get(name), expected)]
        training_mismatches = [f"{name}={training.get(name)!r}" for name, expected in TRAIN_DEFAULTS.items() if not _same(training.get(name), expected)]
        horizons = evaluation.get("horizons", []); full_steps = _first(evaluation, ("full_rollout_steps", "rollout_steps"))
        ok = "argparse" in train_source.lower() and all(f'"--{name}"' in train_source for name in ("config", "data", "output", "device", "epochs", "rollout")); update_check(current_index, ok, "required CLI overrides"); current_index += 1
        ok = not model_mismatches and all(any(alias in model for alias in ((name,) if name != "hidden_dim" else ("hidden_dim", "latent_dim"))) for name in MODEL_DEFAULTS); update_check(current_index, ok, f"mismatches={model_mismatches}"); current_index += 1
        update_check(current_index, not training_mismatches, f"mismatches={training_mismatches}"); current_index += 1
        ok = "torch.optim.adam" in lower and ("lambdalr" in lower or "exponentiallr" in lower) and "scheduler.step" in lower; update_check(current_index, ok, "Adam and per-optimization-step exponential decay"); current_index += 1
        ok = ("normalized_mse" in lower or ("mse_loss" in lower and "normalize(" in lower)) and "loss_nodes" in lower; update_check(current_index, ok, "all and normal_outflow loss modes"); current_index += 1
        ok = "zero_grad" in lower and ".backward" in lower and ("optimizer.step" in lower or "scaler.step(optimizer)" in compact); update_check(current_index, ok, "zero-grad/backward/optimizer-step"); current_index += 1
        ok = ("distributeddataparallel" in lower or "ddp(" in lower) and "autocast" in lower and "gradscaler" in lower; update_check(current_index, ok, "single-device/DDP and AMP"); current_index += 1
        ok = "resume" in lower and "optimizer_state_dict" in lower and "scheduler_state_dict" in lower and "stale_epochs" in lower; update_check(current_index, ok, "full-state resume and early stopping"); current_index += 1
        ok = "best.pt" in lower and "last.pt" in lower and "best_val_loss" in lower and "save_checkpoint" in lower; update_check(current_index, ok, "best by validation normalized MSE and last each epoch"); current_index += 1
        ok = 1 in horizons and 50 in horizons and _same(full_steps, 299) and "evaluate_rollouts" in lower; update_check(current_index, ok, f"horizons={horizons}, full={full_steps}"); current_index += 1
        ok = "normal" in lower and "outflow" in lower and "torch.where" in lower and ("truth_velocity" in lower or "boundary_velocity" in lower); update_check(current_index, ok, "only NORMAL/OUTFLOW are advanced"); current_index += 1
        ok = "velocity_rmse" in lower and "pressure_mse" in lower; update_check(current_index, ok, "velocity RMSE and pressure MSE are separate"); current_index += 1
        ok = all(name in lower for name in ("history.json", "summary.json", "train.log")) and "config" in lower; update_check(current_index, ok, "configuration-bearing history and summary plus file log"); current_index += 1
        ok = "meshgraphnet_cylinderflow_seed0" in str(training.get("output_dir", "")).lower(); update_check(current_index, ok, training.get("output_dir")); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Validate an optional standalone evaluation script; absence is allowed."""
    CHECKPOINT_NAMES = ["eval:optional_absent", "eval:checkpoint", "eval:inference_mode", "eval:rollout_metrics"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        path = next((CANDIDATE_DIR / name for name in ("evaluate.py", "eval.py", "test.py") if (CANDIDATE_DIR / name).is_file()), None)
        if path is None:
            update_check(current_index, True, "rollout evaluation is integrated into train.py"); current_index += 1; current_index = len(CHECKPOINT_NAMES)
        else:
            current_index += 1; lower = _read(path).lower()
            update_check(current_index, "load" in lower and "checkpoint" in lower, path); current_index += 1
            update_check(current_index, ".eval(" in lower and ("no_grad" in lower or "inference_mode" in lower), path); current_index += 1
            update_check(current_index, "rollout" in lower and "velocity" in lower and "pressure" in lower, path); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Inspect the required encode-process-decode graph architecture."""
    CHECKPOINT_NAMES = ["model:meshgraphnet_class", "model:encoders_decoder", "model:fifteen_processors", "model:two_hidden_layer_mlps", "model:feature_dimensions", "model:relu_layernorm", "model:edge_then_node_residual", "model:sum_receiver_aggregation", "model:trainable"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch.nn as nn
        generated_module = _load_module("meshgraphnet_generated_model", CANDIDATE_DIR / "model.py")
        model = _build_model(generated_module).cpu(); processors = _module_list(model)
        node_encoder = getattr(model, "node_encoder", None); edge_encoder = getattr(model, "edge_encoder", None); decoder = getattr(model, "decoder", None)
        source = inspect.getsource(generated_module).lower(); linears = [module for module in model.modules() if isinstance(module, nn.Linear)]; norms = [module for module in model.modules() if isinstance(module, nn.LayerNorm)]
        decoder_norms = [module for module in decoder.modules() if isinstance(module, nn.LayerNorm)] if decoder else []; encoder_linears = [module for module in node_encoder.modules() if isinstance(module, nn.Linear)] if node_encoder else []
        first_block = processors[0] if processors else None; edge_mlp = getattr(first_block, "edge_mlp", None); node_mlp = getattr(first_block, "node_mlp", None)
        edge_linears = [module for module in edge_mlp.modules() if isinstance(module, nn.Linear)] if edge_mlp else []; node_linears = [module for module in node_mlp.modules() if isinstance(module, nn.Linear)] if node_mlp else []; parameter_count = sum(parameter.numel() for parameter in model.parameters())
        update_check(current_index, isinstance(model, nn.Module), type(model).__name__); current_index += 1
        update_check(current_index, node_encoder is not None and edge_encoder is not None and decoder is not None, "node encoder, edge encoder, and node decoder"); current_index += 1
        update_check(current_index, len(processors) == 15, f"processor_blocks={len(processors)}"); current_index += 1
        update_check(current_index, len(encoder_linears) == 3 and len(edge_linears) == 3 and len(node_linears) == 3, f"node_encoder={len(encoder_linears)} edge_update={len(edge_linears)} node_update={len(node_linears)} linears"); current_index += 1
        ok = bool(encoder_linears and edge_linears and node_linears) and encoder_linears[0].in_features == 6 and encoder_linears[-1].out_features == 128 and edge_linears[0].in_features == 3 * 128 and node_linears[0].in_features == 2 * 128 and any(layer.out_features == 3 for layer in linears); update_check(current_index, ok, "6/3 inputs, 128 latent, 3 outputs"); current_index += 1
        update_check(current_index, "relu" in source and len(norms) >= 32 and not decoder_norms, f"layer_norms={len(norms)} decoder_layer_norms={len(decoder_norms)}"); current_index += 1
        ok = "edge" in source and "node" in source and (source.count(" + self.") >= 2 or source.count("+=") >= 2); update_check(current_index, ok, "edge update precedes node update and both are residual"); current_index += 1
        ok = ("index_add" in source or "scatter_add" in source) and ("receiver" in source or "destination" in source); update_check(current_index, ok, "sum aggregation by receiver node"); current_index += 1
        update_check(current_index, parameter_count > 0, f"parameter_elements={parameter_count}"); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Run [N,6],[E,3],[2,E] -> [N,3] on a tiny CPU graph."""
    CHECKPOINT_NAMES = ["forward:return_tensor", "forward:shape", "forward:finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        torch.set_num_threads(1); torch.manual_seed(SEED)
        generated_module = _load_module("meshgraphnet_generated_model", CANDIDATE_DIR / "model.py"); model = _build_model(generated_module).cpu().eval(); node, edge, edge_index, _ = _tiny_graph(torch)
        with torch.no_grad(): output = model(node, edge, edge_index)
        tensor_output = isinstance(output, torch.Tensor); shape = tuple(output.shape) if tensor_output else None
        update_check(current_index, tensor_output, type(output).__name__); current_index += 1
        update_check(current_index, shape == (NODE_COUNT, 3), f"actual={shape} expected={(NODE_COUNT, 3)}"); current_index += 1
        update_check(current_index, tensor_output and bool(torch.isfinite(output).all()), "all output values finite"); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Compare generated and reference forwards under canonical shared weights."""
    CHECKPOINT_NAMES = ["equivalence:parameter_shapes", "equivalence:forward_values"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        torch.set_num_threads(1)
        generated_module = _load_module("meshgraphnet_generated_model", CANDIDATE_DIR / "model.py"); reference_module = _load_module("meshgraphnet_reference_model", REFERENCE_DIR / "model.py")
        generated = _build_model(generated_module).cpu().eval(); reference = _build_model(reference_module).cpu().eval(); generated_shapes = _parameter_shapes(generated); reference_shapes = _parameter_shapes(reference)
        _canonicalize_model(torch, generated); _canonicalize_model(torch, reference); node, edge, edge_index, _ = _tiny_graph(torch)
        with torch.no_grad(): actual = generated(node.clone(), edge.clone(), edge_index.clone()); expected = reference(node.clone(), edge.clone(), edge_index.clone())
        same_shape = isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor) and tuple(actual.shape) == tuple(expected.shape) == (NODE_COUNT, 3); maximum = float((actual - expected).abs().max()) if same_shape else float("inf"); equivalent = same_shape and bool(torch.allclose(actual, expected, atol=FORWARD_ATOL, rtol=FORWARD_RTOL))
        update_check(current_index, generated_shapes == reference_shapes, f"generated={sum(generated_shapes.values())} reference={sum(reference_shapes.values())}"); current_index += 1
        update_check(current_index, equivalent, f"max_abs_diff={maximum:.3e} atol={FORWARD_ATOL} rtol={FORWARD_RTOL}"); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Execute one deterministic CPU optimization step on a tiny graph."""
    CHECKPOINT_NAMES = ["training:parameters", "training:forward_shape", "training:finite_loss", "training:finite_gradients", "training:parameter_updates"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        torch.set_num_threads(1); torch.manual_seed(SEED); generated_module = _load_module("meshgraphnet_generated_model", CANDIDATE_DIR / "model.py"); model = _build_model(generated_module).cpu().train()
        parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if not parameters:
            update_check(current_index, False, "no trainable parameters"); current_index = len(CHECKPOINT_NAMES)
        else:
            update_check(current_index, True, f"{len(parameters)} trainable parameters"); current_index += 1
            before = {name: parameter.detach().clone() for name, parameter in parameters.items()}; optimizer = torch.optim.Adam(parameters.values(), lr=1.0e-4); node, edge, edge_index, target = _tiny_graph(torch); optimizer.zero_grad(set_to_none=True); prediction = model(node, edge, edge_index); loss = torch.nn.functional.mse_loss(prediction, target); loss.backward()
            finite_gradients = {name for name, parameter in parameters.items() if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())}; optimizer.step(); changed = {name for name, parameter in parameters.items() if not torch.equal(before[name], parameter.detach())}; gradient_fraction = len(finite_gradients) / len(parameters); update_fraction = len(changed) / len(parameters)
            update_check(current_index, tuple(prediction.shape) == (NODE_COUNT, 3), tuple(prediction.shape)); current_index += 1
            update_check(current_index, bool(torch.isfinite(loss)), float(loss.detach())); current_index += 1
            update_check(current_index, gradient_fraction >= 0.9, f"{len(finite_gradients)}/{len(parameters)}"); current_index += 1
            update_check(current_index, update_fraction >= 0.8, f"{len(changed)}/{len(parameters)}"); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check artifact declarations and validate materialized outputs when present."""
    CHECKPOINT_NAMES = ["artifacts:declarations", "artifacts:materialized_optional", "artifacts:materialized_if_present"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        source = _project_source(CANDIDATE_DIR, ("train.py", "utils.py")).lower(); config_data = _load_yaml(CANDIDATE_DIR / "config.yaml"); training = _section(config_data, "training", "train"); configured = Path(str(training.get("output_dir", "results/meshgraphnet_cylinderflow_seed0")))
        if configured.is_absolute(): configured = Path(str(configured).lstrip("/"))
        materialized = (CANDIDATE_DIR / configured).resolve()
        ok = all(name in source for name in ("best.pt", "last.pt", "history.json", "summary.json", "train.log")); update_check(current_index, ok, "best/last checkpoints, history, summary, and log"); current_index += 1
        if not materialized.is_dir():
            update_check(current_index, True, "full training artifacts are optional"); current_index += 1; current_index = len(CHECKPOINT_NAMES)
        else:
            current_index += 1; expected = [materialized / name for name in ("best.pt", "last.pt", "history.json", "summary.json", "train.log")]
            try:
                missing = [path.name for path in expected if not path.is_file()]; empty = [path.name for path in expected if path.is_file() and path.stat().st_size == 0]
                for name in ("history.json", "summary.json"):
                    if (materialized / name).is_file(): json.loads(_read(materialized / name))
                ok = not missing and not empty; detail = f"directory={materialized} missing={missing} empty={empty}"
            except Exception as exc:
                ok = False; detail = f"{type(exc).__name__}: {exc}"
            update_check(current_index, ok, detail); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks
