"""V2 adapter for the legacy ESM inference evaluator."""

from pathlib import Path
import sys
import importlib.util
import os
import tempfile
import traceback


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
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["candidate_inference_script_file"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        script_name = Path(config.get("script_name", "run_inference.py"))
        if script_name.is_absolute() or ".." in script_name.parts:
            raise ValueError("script_name must be relative to CANDIDATE_DIR")
        submission_path = (CANDIDATE_DIR / script_name).resolve()
        if not submission_path.is_file():
            raise FileNotFoundError(f"run_inference.py was not found: {submission_path}")
        update_check(current_index, True, str(submission_path))
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "source_has_no_onescience_import",
        "source_calls_eval",
        "source_uses_no_grad_or_inference_mode",
        "required_model_classes_exported",
        "esm_transformer_layer_is_torch_module",
        "esm_backbone_is_torch_module",
        "esm_inference_model_is_torch_module",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import ast
        from torch import nn
        script_name = Path(config.get("script_name", "run_inference.py"))
        if script_name.is_absolute() or ".." in script_name.parts:
            raise ValueError("script_name must be relative to CANDIDATE_DIR")
        submission_path = (CANDIDATE_DIR / script_name).resolve()
        specification = importlib.util.spec_from_file_location("esm_candidate", submission_path)
        if specification is None or specification.loader is None:
            raise ImportError(f"Cannot load Python module from {submission_path}")
        candidate_module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(candidate_module)
        syntax_tree = ast.parse(submission_path.read_text(encoding="utf-8"), filename=str(submission_path))
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import) and "onescience" in {alias.name.split(".")[0] for alias in node.names}:
                raise AssertionError("The generated model must not import onescience")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "onescience":
                raise AssertionError("The generated model must not import onescience")
        update_check(current_index, True, "No onescience import")
        current_index += 1
        calls = [node for node in ast.walk(syntax_tree) if isinstance(node, ast.Call)]
        called_attributes = {call.func.attr for call in calls if isinstance(call.func, ast.Attribute)}
        if "eval" not in called_attributes:
            raise AssertionError("The inference entry point must switch the model to eval mode")
        update_check(current_index, True, "eval is called")
        current_index += 1
        if not {"no_grad", "inference_mode"} & called_attributes:
            raise AssertionError("The inference entry point must use torch.no_grad or torch.inference_mode")
        update_check(current_index, True, "Inference context is used")
        current_index += 1
        required_classes = ("ProteinTokenizer", "ESMTransformerLayer", "ESMBackbone", "ESMInferenceModel")
        missing_classes = [name for name in required_classes if not hasattr(candidate_module, name)]
        if missing_classes:
            raise AssertionError(f"Missing required model classes: {missing_classes}")
        update_check(current_index, True, "All required classes are present")
        current_index += 1
        for class_name in required_classes[1:]:
            model_class = getattr(candidate_module, class_name)
            if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
                raise AssertionError(f"{class_name} must inherit torch.nn.Module")
            update_check(current_index, True, f"{class_name} is a torch module")
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "fasta_tokenizer_primary_equivalence",
        "tokenizer_boundary_batch_equivalence",
        "invalid_protein_input_rejection",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        script_name = Path(config.get("script_name", "run_inference.py"))
        reference_name = Path(config.get("reference_script_name", "run_inference.py"))
        if script_name.is_absolute() or ".." in script_name.parts or reference_name.is_absolute() or ".." in reference_name.parts:
            raise ValueError("script paths must be relative to their contracted directories")
        submission_path = (CANDIDATE_DIR / script_name).resolve()
        reference_path = (REFERENCE_DIR / reference_name).resolve()
        candidate_spec = importlib.util.spec_from_file_location("esm_candidate", submission_path)
        reference_spec = importlib.util.spec_from_file_location("esm_reference", reference_path)
        if candidate_spec is None or candidate_spec.loader is None or reference_spec is None or reference_spec.loader is None:
            raise ImportError("Cannot load candidate or reference ESM module")
        candidate_module = importlib.util.module_from_spec(candidate_spec)
        reference_module = importlib.util.module_from_spec(reference_spec)
        candidate_spec.loader.exec_module(candidate_module)
        reference_spec.loader.exec_module(reference_module)
        with tempfile.TemporaryDirectory(prefix="esm_eval_") as temporary_dir:
            work_root = Path(temporary_dir)
            fasta_path = work_root / "data_processing.fasta"
            fasta_path.write_text(">protein one\nmk\nTW\n\n>protein-two\nACDE\nFG\n", encoding="utf-8")
            reference_records = reference_module.read_fasta(fasta_path)
            if reference_records != [("protein one", "mkTW"), ("protein-two", "ACDEFG")]:
                raise AssertionError(f"Reference FASTA parsing mismatch: {reference_records!r}")
            reference_tokenizer = reference_module.ProteinTokenizer()
            candidate_tokenizer = candidate_module.ProteinTokenizer()
            if candidate_tokenizer.token_to_index != reference_tokenizer.token_to_index:
                raise AssertionError("Tokenizer vocabulary or token order does not match reference")
            for attribute_name in ("pad_index", "cls_index", "eos_index", "unknown_index"):
                if getattr(candidate_tokenizer, attribute_name) != getattr(reference_tokenizer, attribute_name):
                    raise AssertionError(f"Tokenizer {attribute_name} does not match reference")
            reference_tokens = reference_tokenizer.encode(["mkTW", "ACDEFG"])
            candidate_tokens = candidate_tokenizer.encode(["mkTW", "ACDEFG"])
            if not isinstance(candidate_tokens, torch.Tensor) or candidate_tokens.dtype != torch.long or tuple(candidate_tokens.shape) != (2, 8):
                raise AssertionError("Tokenizer primary output contract failed")
            torch.testing.assert_close(candidate_tokens, reference_tokens, rtol=0.0, atol=0.0)
            if int(candidate_tokens[0, 0]) != candidate_tokenizer.cls_index or int(candidate_tokens[0, 5]) != candidate_tokenizer.eos_index or not candidate_tokens[0, 6:].eq(candidate_tokenizer.pad_index).all():
                raise AssertionError("Tokenizer special-token placement failed")
            update_check(current_index, True, "Primary FASTA and tokenizer behavior matches")
            current_index += 1
            for sequences in ("A", " acdefghiklmnpqrstvwy ", ["M", "AC", "WYV"], ["A" * 17, "C" * 3, "MKTWQ"], [reference_module.AMINO_ACIDS, reference_module.AMINO_ACIDS[::-1]]):
                reference_tokens = reference_tokenizer.encode(sequences)
                candidate_tokens = candidate_tokenizer.encode(sequences)
                torch.testing.assert_close(candidate_tokens, reference_tokens, rtol=0.0, atol=0.0)
                if candidate_tokens.dtype != torch.long or candidate_tokens.ndim != 2 or not candidate_tokens[:, 0].eq(candidate_tokenizer.cls_index).all():
                    raise AssertionError("Boundary tokenizer tensor contract failed")
                normalized = [sequences] if isinstance(sequences, str) else sequences
                for row, sequence in enumerate(sequence.strip().upper() for sequence in normalized):
                    eos_column = len(sequence) + 1
                    if int(candidate_tokens[row, eos_column]) != candidate_tokenizer.eos_index or not candidate_tokens[row, eos_column + 1:].eq(candidate_tokenizer.pad_index).all():
                        raise AssertionError("Boundary tokenizer EOS or padding contract failed")
            for residue in reference_module.AMINO_ACIDS:
                if candidate_tokenizer.get_idx(residue) != reference_tokenizer.get_idx(residue):
                    raise AssertionError(f"Tokenizer residue index mismatch for {residue}")
            update_check(current_index, True, "Boundary tokenizer batches match")
            current_index += 1
            for description, sequences in (("empty protein batch", []), ("empty protein sequence", [""]), ("unsupported amino acid", ["ACDX"])):
                try:
                    candidate_tokenizer.encode(sequences)
                except ValueError:
                    pass
                except Exception as error:
                    raise AssertionError(f"{description} raised {type(error).__name__}, expected ValueError") from error
                else:
                    raise AssertionError(f"{description} did not raise ValueError")
            update_check(current_index, True, "Invalid protein inputs are rejected")
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "transformer_padding_mask_component_equivalence",
        "masked_key_value_invariance",
        "backbone_padded_batch_equivalence",
        "fixed_input_forward_equivalence",
        "intermediate_forward_activation_equivalence",
        "six_batch_boundary_length_equivalence",
        "batched_vs_single_inference_isolation",
        "inference_determinism_and_state_integrity",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        script_name = Path(config.get("script_name", "run_inference.py"))
        reference_name = Path(config.get("reference_script_name", "run_inference.py"))
        if script_name.is_absolute() or ".." in script_name.parts or reference_name.is_absolute() or ".." in reference_name.parts:
            raise ValueError("script paths must be relative to their contracted directories")
        submission_path = (CANDIDATE_DIR / script_name).resolve()
        reference_path = (REFERENCE_DIR / reference_name).resolve()
        candidate_spec = importlib.util.spec_from_file_location("esm_candidate", submission_path)
        reference_spec = importlib.util.spec_from_file_location("esm_reference", reference_path)
        if candidate_spec is None or candidate_spec.loader is None or reference_spec is None or reference_spec.loader is None:
            raise ImportError("Cannot load candidate or reference ESM module")
        candidate_module = importlib.util.module_from_spec(candidate_spec)
        reference_module = importlib.util.module_from_spec(reference_spec)
        candidate_spec.loader.exec_module(candidate_module)
        reference_spec.loader.exec_module(reference_module)
        torch.manual_seed(20260729)
        reference_layer = reference_module.ESMTransformerLayer(32, 4, 64).eval()
        candidate_layer = candidate_module.ESMTransformerLayer(32, 4, 64).eval()
        candidate_layer.load_state_dict(reference_layer.state_dict(), strict=True)
        layer_input = torch.linspace(-1.0, 1.0, steps=2 * 7 * 32).reshape(2, 7, 32)
        padding_mask = torch.tensor([[False] * 7, [False, False, False, False, True, True, True]], dtype=torch.bool)
        with torch.no_grad():
            reference_layer_output = reference_layer(layer_input, padding_mask)
            candidate_layer_output = candidate_layer(layer_input, padding_mask)
        if tuple(candidate_layer_output.shape) != tuple(layer_input.shape):
            raise AssertionError("ESMTransformerLayer must preserve batch, token, and embedding dimensions")
        torch.testing.assert_close(candidate_layer_output, reference_layer_output, rtol=1e-6, atol=1e-7)
        update_check(current_index, True, "Transformer layer matches reference")
        current_index += 1
        torch.manual_seed(20260731)
        reference_layer = reference_module.ESMTransformerLayer(32, 4, 64).eval()
        candidate_layer = candidate_module.ESMTransformerLayer(32, 4, 64).eval()
        candidate_layer.load_state_dict(reference_layer.state_dict(), strict=True)
        inputs = torch.linspace(-0.9, 0.9, steps=2 * 8 * 32).reshape(2, 8, 32)
        padding_mask = torch.tensor([[False] * 8, [False, False, False, False, False, True, True, True]], dtype=torch.bool)
        perturbed_inputs = inputs.clone()
        perturbed_inputs[1, 5:] = torch.linspace(-100.0, 100.0, steps=3 * 32).reshape(3, 32)
        with torch.no_grad():
            reference_output = reference_layer(inputs, padding_mask)
            candidate_output = candidate_layer(inputs, padding_mask)
            reference_perturbed = reference_layer(perturbed_inputs, padding_mask)
            candidate_perturbed = candidate_layer(perturbed_inputs, padding_mask)
        torch.testing.assert_close(candidate_output, reference_output, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(candidate_perturbed, reference_perturbed, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(candidate_output[1, :5], candidate_perturbed[1, :5], rtol=1e-6, atol=1e-7)
        update_check(current_index, True, "Masked positions do not affect valid outputs")
        current_index += 1
        with tempfile.TemporaryDirectory(prefix="esm_eval_") as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "reference_weights.pt"
            torch.manual_seed(20260729)
            torch.save(reference_module.ESMInferenceModel().state_dict(), checkpoint_path)
            try:
                saved_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                saved_state = torch.load(checkpoint_path, map_location="cpu")
            reference_model = reference_module.ESMInferenceModel()
            candidate_model = candidate_module.ESMInferenceModel()
            if set(reference_model.state_dict()) != set(candidate_model.state_dict()):
                raise AssertionError("Model state layout mismatch")
            reference_model.load_state_dict(saved_state, strict=True)
            candidate_model.load_state_dict(saved_state, strict=True)
            reference_model.eval()
            candidate_model.eval()
            reference_tokenizer = reference_module.ProteinTokenizer()
            candidate_tokenizer = candidate_module.ProteinTokenizer()
            sequences = ["MKTW", "ACDEFGHIK"]
            with torch.no_grad():
                reference_backbone = reference_model.backbone(reference_tokenizer.encode(sequences))
                candidate_backbone = candidate_model.backbone(candidate_tokenizer.encode(sequences))
            if tuple(candidate_backbone.shape) != (2, 11, 32) or not torch.isfinite(candidate_backbone).all():
                raise AssertionError("ESMBackbone output contract failed")
            torch.testing.assert_close(candidate_backbone, reference_backbone, rtol=1e-6, atol=1e-7)
            update_check(current_index, True, "Backbone matches reference")
            current_index += 1
            sequences = ["MKTW", "ACDEFG"]
            reference_tokens = reference_tokenizer.encode(sequences)
            candidate_tokens = candidate_tokenizer.encode(sequences)
            torch.testing.assert_close(candidate_tokens, reference_tokens, rtol=0.0, atol=0.0)
            with torch.no_grad():
                reference_outputs = reference_model(reference_tokens)
                candidate_outputs = candidate_model(candidate_tokens)
            expected_shapes = {"representations": (2, 8, 32), "logits": (2, 8, 25), "coordinates": (2, 8, 3)}
            if not isinstance(candidate_outputs, dict) or set(candidate_outputs) != set(expected_shapes):
                raise AssertionError("ESMInferenceModel.forward output contract failed")
            for name, shape in expected_shapes.items():
                if not isinstance(candidate_outputs[name], torch.Tensor) or tuple(candidate_outputs[name].shape) != shape or candidate_outputs[name].dtype != reference_outputs[name].dtype or not torch.isfinite(candidate_outputs[name]).all():
                    raise AssertionError(f"{name} output contract failed")
                torch.testing.assert_close(candidate_outputs[name], reference_outputs[name], rtol=1e-6, atol=1e-7)
            update_check(current_index, True, "Fixed forward matches reference")
            current_index += 1
            reference_modules = [reference_model.backbone.token_embedding, reference_model.backbone.position_embedding, reference_model.backbone.layers[0], reference_model.backbone.layers[1], reference_model.backbone.final_norm, reference_model.lm_head, reference_model.coordinate_head]
            candidate_modules = [candidate_model.backbone.token_embedding, candidate_model.backbone.position_embedding, candidate_model.backbone.layers[0], candidate_model.backbone.layers[1], candidate_model.backbone.final_norm, candidate_model.lm_head, candidate_model.coordinate_head]
            reference_activations, candidate_activations = [], []
            reference_handles = [module.register_forward_hook(lambda _m, _i, output: reference_activations.append(output.detach().cpu())) for module in reference_modules]
            candidate_handles = [module.register_forward_hook(lambda _m, _i, output: candidate_activations.append(output.detach().cpu())) for module in candidate_modules]
            with torch.no_grad():
                reference_model(reference_tokenizer.encode(["MKTWQ", "ACDEFGHIK"]))
                candidate_model(candidate_tokenizer.encode(["MKTWQ", "ACDEFGHIK"]))
            for handle in [*reference_handles, *candidate_handles]:
                handle.remove()
            if len(candidate_activations) != len(reference_activations):
                raise AssertionError("ESM forward hook stages do not match")
            for candidate_activation, reference_activation in zip(candidate_activations, reference_activations):
                if tuple(candidate_activation.shape) != tuple(reference_activation.shape):
                    raise AssertionError("Intermediate activation shape mismatch")
                torch.testing.assert_close(candidate_activation, reference_activation, rtol=1e-6, atol=1e-7)
            update_check(current_index, True, "Intermediate activations match")
            current_index += 1
            longest_sequence = (reference_module.AMINO_ACIDS * 7)[:126]
            for sequences in (["A"], ["AC", "W"], ["MKTWQ", "ACDEFGHIK", "Y"], [reference_module.AMINO_ACIDS, reference_module.AMINO_ACIDS[::-1]], [(reference_module.AMINO_ACIDS * 2)[:32]], [longest_sequence]):
                with torch.no_grad():
                    reference_outputs = reference_model(reference_tokenizer.encode(sequences))
                    candidate_outputs = candidate_model(candidate_tokenizer.encode(sequences))
                for name in expected_shapes:
                    torch.testing.assert_close(candidate_outputs[name], reference_outputs[name], rtol=1e-6, atol=1e-7)
            update_check(current_index, True, "Boundary forward batches match")
            current_index += 1
            for tokens in (torch.zeros(1, 2, 3, dtype=torch.long), torch.zeros(1, 129, dtype=torch.long)):
                try:
                    candidate_model(tokens)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid ESM token input did not raise ValueError")
            sequences = ["MKTW", "ACDEFGHIK", "W"]
            with torch.no_grad():
                candidate_batch_outputs = candidate_model(candidate_tokenizer.encode(sequences))
            for batch_index, sequence in enumerate(sequences):
                with torch.no_grad():
                    single_outputs = candidate_model(candidate_tokenizer.encode(sequence))
                for name in expected_shapes:
                    torch.testing.assert_close(candidate_batch_outputs[name][batch_index, :len(sequence) + 2], single_outputs[name][0], rtol=2e-6, atol=2e-7)
            update_check(current_index, True, "Batch isolation holds")
            current_index += 1
            if any(module.training for module in candidate_model.modules()):
                raise AssertionError("Model remained in training mode")
            state_before = {name: value.detach().clone() for name, value in candidate_model.state_dict().items()}
            tokens = candidate_tokenizer.encode(["ACDEFGHIKLMN", "WYVTSRQPNMLK"])
            with torch.no_grad():
                outputs_one = candidate_model(tokens)
                outputs_two = candidate_model(tokens)
                outputs_three = candidate_model(tokens)
            for name in expected_shapes:
                if outputs_two[name].requires_grad or outputs_three[name].requires_grad:
                    raise AssertionError("Inference output tracks gradients")
                torch.testing.assert_close(outputs_two[name], outputs_one[name], rtol=0.0, atol=0.0)
                torch.testing.assert_close(outputs_three[name], outputs_one[name], rtol=0.0, atol=0.0)
            for name, expected in state_before.items():
                torch.testing.assert_close(candidate_model.state_dict()[name], expected, rtol=0.0, atol=0.0)
            update_check(current_index, True, "Inference is deterministic and preserves state")
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["raw_strict_reference_weight_loading", "wrapped_strict_reference_weight_loading", "incomplete_checkpoint_rejection"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        script_name = Path(config.get("script_name", "run_inference.py"))
        reference_name = Path(config.get("reference_script_name", "run_inference.py"))
        if script_name.is_absolute() or ".." in script_name.parts or reference_name.is_absolute() or ".." in reference_name.parts:
            raise ValueError("script paths must be relative to their contracted directories")
        submission_path = (CANDIDATE_DIR / script_name).resolve()
        reference_path = (REFERENCE_DIR / reference_name).resolve()
        candidate_spec = importlib.util.spec_from_file_location("esm_candidate", submission_path)
        reference_spec = importlib.util.spec_from_file_location("esm_reference", reference_path)
        if candidate_spec is None or candidate_spec.loader is None or reference_spec is None or reference_spec.loader is None:
            raise ImportError("Cannot load candidate or reference ESM module")
        candidate_module = importlib.util.module_from_spec(candidate_spec)
        reference_module = importlib.util.module_from_spec(reference_spec)
        candidate_spec.loader.exec_module(candidate_module)
        reference_spec.loader.exec_module(reference_module)
        with tempfile.TemporaryDirectory(prefix="esm_eval_") as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "reference_weights.pt"
            torch.manual_seed(20260729)
            torch.save(reference_module.ESMInferenceModel().state_dict(), checkpoint_path)
            try:
                saved_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                saved_state = torch.load(checkpoint_path, map_location="cpu")
            candidate_module.ESMInferenceModel().load_state_dict(saved_state, strict=True)
            update_check(current_index, True, "Raw state_dict loads strictly")
            current_index += 1
            wrapped_checkpoint_path = Path(temporary_dir) / "wrapped_reference_weights.pt"
            torch.save({"model_state_dict": saved_state}, wrapped_checkpoint_path)
            try:
                wrapped_state = torch.load(wrapped_checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                wrapped_state = torch.load(wrapped_checkpoint_path, map_location="cpu")
            candidate_module.ESMInferenceModel().load_state_dict(wrapped_state["model_state_dict"], strict=True)
            update_check(current_index, True, "Wrapped model_state_dict loads strictly")
            current_index += 1
            invalid_state = dict(saved_state)
            invalid_state.pop(next(iter(invalid_state)))
            try:
                candidate_module.ESMInferenceModel().load_state_dict(invalid_state, strict=True)
            except RuntimeError as error:
                if not str(error):
                    raise AssertionError("Incomplete checkpoint failed without an error message")
            else:
                raise AssertionError("Incomplete checkpoint unexpectedly loaded")
            update_check(current_index, True, "Incomplete checkpoint is rejected")
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "wrapped_checkpoint_extract_artifact_equivalence",
        "wrapped_checkpoint_fold_artifact_equivalence",
        "raw_checkpoint_extract_artifact_equivalence",
        "empty_fasta_rejection",
        "headerless_fasta_rejection",
        "unsupported_residue_fasta_rejection",
        "variant_artifact_equivalence",
        "invalid_mutation_rejection",
        "invalid_checkpoint_cli_rejection",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import csv
        import subprocess
        import torch
        script_name = Path(config.get("script_name", "run_inference.py"))
        reference_name = Path(config.get("reference_script_name", "run_inference.py"))
        if script_name.is_absolute() or ".." in script_name.parts or reference_name.is_absolute() or ".." in reference_name.parts:
            raise ValueError("script paths must be relative to their contracted directories")
        submission_path = (CANDIDATE_DIR / script_name).resolve()
        reference_path = (REFERENCE_DIR / reference_name).resolve()
        reference_spec = importlib.util.spec_from_file_location("esm_reference", reference_path)
        if reference_spec is None or reference_spec.loader is None:
            raise ImportError(f"Cannot load Python module from {reference_path}")
        reference_module = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference_module)
        with tempfile.TemporaryDirectory(prefix="esm_eval_") as temporary_dir:
            work_root = Path(temporary_dir)
            torch.manual_seed(20260729)
            raw_checkpoint_path = work_root / "reference_weights.pt"
            torch.save(reference_module.ESMInferenceModel().state_dict(), raw_checkpoint_path)
            try:
                saved_state = torch.load(raw_checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                saved_state = torch.load(raw_checkpoint_path, map_location="cpu")
            wrapped_checkpoint_path = work_root / "wrapped_reference_weights.pt"
            torch.save({"model_state_dict": saved_state}, wrapped_checkpoint_path)
            invalid_state = dict(saved_state)
            invalid_state.pop(next(iter(invalid_state)))
            invalid_checkpoint_path = work_root / "invalid_reference_weights.pt"
            torch.save(invalid_state, invalid_checkpoint_path)
            fasta_path = work_root / "proteins.fasta"
            fasta_path.write_text(">protein one\nMKTW\n>protein-two\nACDEFG\n", encoding="utf-8")
            def run(script_path, arguments):
                completed = subprocess.run([sys.executable, str(script_path), *arguments], cwd=script_path.parent, capture_output=True, text=True, timeout=90, check=False)
                if completed.returncode != 0:
                    raise AssertionError(f"Inference CLI failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
            reference_extract = work_root / "reference_extract"
            candidate_extract = work_root / "candidate_extract"
            extract_args = ["--task", "extract", "--checkpoint", str(wrapped_checkpoint_path), "--fasta", str(fasta_path), "--device", "cpu"]
            run(reference_path, [*extract_args, "--output", str(reference_extract)])
            run(submission_path, [*extract_args, "--output", str(candidate_extract)])
            reference_files = {path.name for path in reference_extract.glob("*.pt")}
            candidate_files = {path.name for path in candidate_extract.glob("*.pt")}
            if candidate_files != reference_files or len(candidate_files) != 2:
                raise AssertionError("Representation output file set does not match reference")
            for filename in sorted(reference_files):
                try:
                    reference = torch.load(reference_extract / filename, map_location="cpu", weights_only=True)
                    candidate = torch.load(candidate_extract / filename, map_location="cpu", weights_only=True)
                except TypeError:
                    reference = torch.load(reference_extract / filename, map_location="cpu")
                    candidate = torch.load(candidate_extract / filename, map_location="cpu")
                if not isinstance(candidate, dict) or set(candidate) != {"label", "representations", "mean_representations"} or candidate["label"] != reference["label"]:
                    raise AssertionError(f"Unexpected representation fields in {filename}")
                for field in ("representations", "mean_representations"):
                    if not isinstance(candidate[field], torch.Tensor) or candidate[field].dtype != reference[field].dtype or tuple(candidate[field].shape) != tuple(reference[field].shape) or not torch.isfinite(candidate[field]).all():
                        raise AssertionError(f"Invalid {field} in {filename}")
                    torch.testing.assert_close(candidate[field], reference[field], rtol=1e-6, atol=1e-7)
            update_check(current_index, True, "Wrapped checkpoint extract artifacts match")
            current_index += 1
            reference_fold = work_root / "reference_fold"
            candidate_fold = work_root / "candidate_fold"
            fold_args = ["--task", "fold", "--checkpoint", str(wrapped_checkpoint_path), "--fasta", str(fasta_path), "--device", "cpu"]
            run(reference_path, [*fold_args, "--output", str(reference_fold)])
            run(submission_path, [*fold_args, "--output", str(candidate_fold)])
            reference_files = {path.name for path in reference_fold.glob("*.pdb")}
            candidate_files = {path.name for path in candidate_fold.glob("*.pdb")}
            if candidate_files != reference_files or len(candidate_files) != 2:
                raise AssertionError("PDB output file set does not match reference")
            for filename in sorted(reference_files):
                reference_text = (reference_fold / filename).read_text(encoding="utf-8")
                candidate_text = (candidate_fold / filename).read_text(encoding="utf-8")
                if candidate_text != reference_text or "ATOM" not in candidate_text or not candidate_text.rstrip().endswith("END"):
                    raise AssertionError(f"Invalid PDB content in {filename}")
            update_check(current_index, True, "Wrapped checkpoint fold artifacts match")
            current_index += 1
            raw_extract = work_root / "candidate_raw_extract"
            run(submission_path, ["--task", "extract", "--checkpoint", str(raw_checkpoint_path), "--fasta", str(fasta_path), "--device", "cpu", "--output", str(raw_extract)])
            for filename in sorted({path.name for path in reference_extract.glob("*.pt")}):
                try:
                    reference = torch.load(reference_extract / filename, map_location="cpu", weights_only=True)
                    candidate = torch.load(raw_extract / filename, map_location="cpu", weights_only=True)
                except TypeError:
                    reference = torch.load(reference_extract / filename, map_location="cpu")
                    candidate = torch.load(raw_extract / filename, map_location="cpu")
                torch.testing.assert_close(candidate["representations"], reference["representations"], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(candidate["mean_representations"], reference["mean_representations"], rtol=1e-6, atol=1e-7)
            update_check(current_index, True, "Raw checkpoint extract artifacts match")
            current_index += 1
            for name, content in (("empty", ""), ("headerless", "ACDE\n"), ("unsupported_residue", ">invalid\nACDX\n")):
                invalid_fasta = work_root / f"{name}.fasta"
                invalid_fasta.write_text(content, encoding="utf-8")
                completed = subprocess.run([sys.executable, str(submission_path), "--task", "extract", "--checkpoint", str(wrapped_checkpoint_path), "--fasta", str(invalid_fasta), "--output", str(work_root / f"invalid_{name}"), "--device", "cpu"], cwd=submission_path.parent, capture_output=True, text=True, timeout=90, check=False)
                if completed.returncode == 0 or not (completed.stdout.strip() or completed.stderr.strip()):
                    raise AssertionError(f"{name} FASTA validation did not fail clearly")
                update_check(current_index, True, f"{name} FASTA is rejected")
                current_index += 1
            dms_input = work_root / "variants.csv"
            dms_input.write_text("mutant,assay\nA1C,first\nD3A,second\n", encoding="utf-8")
            reference_csv = work_root / "reference_variants.csv"
            candidate_csv = work_root / "candidate_variants.csv"
            variant_args = ["--task", "variant", "--checkpoint", str(wrapped_checkpoint_path), "--sequence", "ACDE", "--dms-input", str(dms_input), "--mutation-col", "mutant", "--offset-idx", "1", "--device", "cpu"]
            run(reference_path, [*variant_args, "--output", str(reference_csv)])
            run(submission_path, [*variant_args, "--output", str(candidate_csv)])
            with reference_csv.open("r", encoding="utf-8", newline="") as handle:
                reference_rows = list(csv.DictReader(handle))
            with candidate_csv.open("r", encoding="utf-8", newline="") as handle:
                candidate_rows = list(csv.DictReader(handle))
            if len(candidate_rows) != len(reference_rows) or len(candidate_rows) != 2:
                raise AssertionError("Variant output row count does not match reference")
            for reference, candidate in zip(reference_rows, candidate_rows):
                if set(candidate) != set(reference):
                    raise AssertionError("Variant output columns do not match reference")
                for field in set(reference) - {"esm_score"}:
                    if candidate[field] != reference[field]:
                        raise AssertionError(f"Variant field {field!r} changed")
                if abs(float(candidate["esm_score"]) - float(reference["esm_score"])) > 1e-6:
                    raise AssertionError(f"Variant score mismatch for {candidate['mutant']}")
            update_check(current_index, True, "Variant artifacts match")
            current_index += 1
            invalid_dms = work_root / "invalid_variants.csv"
            invalid_dms.write_text("mutant,assay\nW1C,invalid\n", encoding="utf-8")
            for arguments in (["--task", "variant", "--checkpoint", str(wrapped_checkpoint_path), "--sequence", "ACDE", "--dms-input", str(invalid_dms), "--mutation-col", "mutant", "--offset-idx", "1", "--output", str(work_root / "invalid_variant_output"), "--device", "cpu"], ["--task", "extract", "--checkpoint", str(invalid_checkpoint_path), "--fasta", str(fasta_path), "--output", str(work_root / "invalid_checkpoint_output"), "--device", "cpu"]):
                completed = subprocess.run([sys.executable, str(submission_path), *arguments], cwd=submission_path.parent, capture_output=True, text=True, timeout=90, check=False)
                if completed.returncode == 0 or not (completed.stdout.strip() or completed.stderr.strip()):
                    raise AssertionError("Required CLI failure did not fail clearly")
                update_check(current_index, True, "Required CLI failure is reported")
                current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks
