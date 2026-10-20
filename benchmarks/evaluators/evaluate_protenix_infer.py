"""V2 adapter for the legacy Protenix inference evaluator."""

import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable

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

OFFICIAL_CONFIG_PATH = REFERENCE_DIR / "inference_config.yaml"
OFFICIAL_REFERENCE_OUTPUT_PATH = REFERENCE_DIR / "protenix_infer_reference.json"
OFFICIAL_CHECKPOINT_PATH = Path(
    "/public/share/sugonhpcapp01/onestore/onemodels/Protenix/model_v0.5.0.pt"
)
PREDICTION_FIELDS = ("coordinate", "plddt", "pae", "pde", "resolved", "contact_probs")
DETERMINISTIC_SEED = 101
STOCHASTIC_SEEDS = (101, 202, 303)
NUMERIC_RTOL = 1e-4
NUMERIC_ATOL = 1e-5
EVALUATION_N_CYCLE = 1
EVALUATION_N_MODEL_SEED = 1
EVALUATION_N_SAMPLE = 1
EVALUATION_N_STEP = 2
EVALUATION_DIFFUSION_BATCH_SIZE = 48
EVALUATION_TRAIN_NOISE_SAMPLER = {
    "p_mean": -1.2,
    "p_std": 1.5,
    "sigma_data": 16.0,
}


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check required candidate and reference artifacts."""
    CHECKPOINT_NAMES = [
        "reference_config_exists",
        "candidate_config_exists",
        "official_checkpoint_exists",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission_config_path = CANDIDATE_DIR / "inference_config.yaml"
        checkpoint_path = OFFICIAL_CHECKPOINT_PATH
        if not OFFICIAL_CONFIG_PATH.is_file():
            raise FileNotFoundError(f"Official configuration does not exist: {OFFICIAL_CONFIG_PATH}")
        update_check(current_index, True, f"Found {OFFICIAL_CONFIG_PATH}")
        current_index += 1
        if not submission_config_path.is_file():
            raise FileNotFoundError(f"Submission configuration does not exist: {submission_config_path}")
        update_check(current_index, True, f"Found {submission_config_path}")
        current_index += 1
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Official checkpoint does not exist: {checkpoint_path}")
        update_check(current_index, True, f"Found {checkpoint_path}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_config(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check legacy YAML parsing and reference configuration materialization."""
    CHECKPOINT_NAMES = [
        "reference_inference_config_mapping",
        "candidate_config_mapping",
        "reference_config_materialization",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission_config_path = CANDIDATE_DIR / "inference_config.yaml"
        checkpoint_path = OFFICIAL_CHECKPOINT_PATH
        standard_config_payload = _legacy.load_config_payload(
            OFFICIAL_CONFIG_PATH, require_inference_root=True
        )
        update_check(current_index, True, "Reference YAML is a mapping with an inference mapping")
        current_index += 1
        submission_config_payload = _legacy.load_config_payload(submission_config_path)
        update_check(current_index, True, "Candidate YAML is a mapping")
        current_index += 1
        import torch

        standard_configs = _legacy.materialize_inference_config(
            standard_config_payload, checkpoint_path
        )
        del torch, standard_configs
        update_check(
            current_index,
            True,
            "Reference configuration received the legacy fixed evaluation settings",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the public Protenix import and supported construction interface."""
    CHECKPOINT_NAMES = [
        "candidate_protenix_module_import",
        "public_protenix_class_name",
        "supported_constructor_configuration",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission_config_path = CANDIDATE_DIR / "inference_config.yaml"
        checkpoint_path = OFFICIAL_CHECKPOINT_PATH
        submission_config_payload = _legacy.load_config_payload(submission_config_path)
        Protenix = _legacy.import_protenix_class(CANDIDATE_DIR)
        update_check(current_index, True, "Imported protenix_infer.Protenix from CANDIDATE_DIR")
        current_index += 1
        if Protenix.__name__ != "Protenix":
            raise AssertionError(
                f"submission public class must be Protenix, got {Protenix.__name__}"
            )
        update_check(current_index, True, "Public class name is Protenix")
        current_index += 1
        model, construction_metrics = _legacy.instantiate_submission_model(
            Protenix,
            submission_config_path,
            submission_config_payload,
            checkpoint_path,
        )
        model.eval()
        update_check(
            current_index,
            True,
            f"Constructed Protenix through {construction_metrics['config_interface']}",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check legacy semantic and name-independent parameter structure equivalence."""
    CHECKPOINT_NAMES = [
        "reference_model_construction",
        "reference_semantic_spec_completeness",
        "reference_parameter_profile",
        "candidate_model_construction",
        "candidate_nonzero_parameter_counts",
        "semantic_component_presence",
        "semantic_depths",
        "semantic_channels",
        "semantic_attention_heads",
        "semantic_atom_windows",
        "semantic_inference_hyperparameters",
        "global_parameter_profile_equivalence",
        "stage_parameter_profile_equivalence",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission_config_path = CANDIDATE_DIR / "inference_config.yaml"
        checkpoint_path = OFFICIAL_CHECKPOINT_PATH
        standard_config_payload = _legacy.load_config_payload(
            OFFICIAL_CONFIG_PATH, require_inference_root=True
        )
        submission_config_payload = _legacy.load_config_payload(submission_config_path)
        standard_configs = _legacy.materialize_inference_config(
            standard_config_payload, checkpoint_path
        )

        standard_model = None
        standard_class = None
        standard_components = None
        try:
            standard_class = _legacy.import_protenix_class(REFERENCE_DIR)
            standard_model = standard_class(standard_configs)
            standard_runtime_metrics = _legacy.apply_evaluation_runtime_overrides(standard_model)
            standard_model.eval()
            standard_semantic_spec, standard_component_discovery = (
                _legacy.collect_semantic_model_spec(standard_model)
            )
            update_check(current_index, True, "Constructed reference Protenix model")
            current_index += 1
            standard_missing_semantics = _legacy.missing_semantic_paths(standard_semantic_spec)
            if standard_missing_semantics:
                raise AssertionError(
                    "Unable to extract required semantic fields from reference model: "
                    f"{standard_missing_semantics}"
                )
            update_check(current_index, True, "Reference semantic specification is complete")
            current_index += 1
            standard_components, _ = _legacy.discover_semantic_components(standard_model)
            standard_parameter_profiles = _legacy.collect_model_parameter_profiles(
                standard_model, standard_components
            )
            update_check(current_index, True, "Collected reference global and stage parameter profiles")
            current_index += 1
        finally:
            standard_model = None
            standard_class = None
            standard_components = None
            _legacy.unload_modules_from_root(REFERENCE_DIR)
            gc.collect()

        Protenix = _legacy.import_protenix_class(CANDIDATE_DIR)
        if Protenix.__name__ != "Protenix":
            raise AssertionError(
                f"submission public class must be Protenix, got {Protenix.__name__}"
            )
        model, construction_metrics = _legacy.instantiate_submission_model(
            Protenix,
            submission_config_path,
            submission_config_payload,
            checkpoint_path,
        )
        submission_runtime_metrics = _legacy.apply_evaluation_runtime_overrides(model)
        model.eval()
        update_check(
            current_index,
            True,
            f"Constructed candidate model through {construction_metrics['config_interface']}",
        )
        current_index += 1
        submission_semantic_spec, submission_component_discovery = (
            _legacy.collect_semantic_model_spec(model)
        )
        submission_components, _ = _legacy.discover_semantic_components(model)
        submission_parameter_profiles = _legacy.collect_model_parameter_profiles(
            model, submission_components
        )
        parameter_count = submission_parameter_profiles["global"]["summary"]["total_numel"]
        trainable_parameter_count = submission_parameter_profiles["global"]["summary"]["trainable_numel"]
        if parameter_count <= 0 or trainable_parameter_count <= 0:
            raise AssertionError("Protenix parameter count or trainable parameter count is zero")
        update_check(
            current_index,
            True,
            f"total_numel={parameter_count}, trainable_numel={trainable_parameter_count}",
        )
        current_index += 1

        semantic_model_comparison = _legacy.compare_semantic_model_specs(
            standard_semantic_spec, submission_semantic_spec
        )
        for section in (
            "component_presence",
            "depths",
            "channels",
            "attention_heads",
            "atom_windows",
            "inference_hyperparameters",
        ):
            section_metrics = semantic_model_comparison["section_counts"][section]
            if section_metrics["differences"]:
                raise AssertionError(
                    f"Semantic section {section} differs: {section_metrics['differences']} fields"
                )
            update_check(current_index, True, f"Semantic section {section} matches")
            current_index += 1

        parameter_profile_comparison = _legacy.compare_model_parameter_profiles(
            standard_parameter_profiles, submission_parameter_profiles
        )
        global_metrics = parameter_profile_comparison["section_counts"]["global"]
        if global_metrics["differences"]:
            raise AssertionError(
                f"Global parameter profile differs: {global_metrics['differences']} fields"
            )
        update_check(current_index, True, "Global parameter profile matches")
        current_index += 1
        stage_metrics = parameter_profile_comparison["section_counts"]["stages"]
        if stage_metrics["differences"]:
            raise AssertionError(
                f"Stage parameter profiles differ: {stage_metrics['differences']} fields"
            )
        update_check(current_index, True, "All stage parameter profiles match")
        current_index += 1
        del standard_runtime_metrics, submission_runtime_metrics, standard_component_discovery
        del submission_component_discovery, semantic_model_comparison
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check legacy official checkpoint loading with its explicit mapping rules."""
    CHECKPOINT_NAMES = [
        "candidate_model_for_checkpoint_loading",
        "model_device_transfer",
        "all_parameters_on_selected_device",
        "checkpoint_payload_mapping",
        "checkpoint_model_state_mapping",
        "checkpoint_module_prefix_normalization",
        "official_parameter_key_mapping",
        "official_parameter_shape_and_dtype_compatibility",
        "strict_checkpoint_loading",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        submission_config_path = CANDIDATE_DIR / "inference_config.yaml"
        checkpoint_path = OFFICIAL_CHECKPOINT_PATH
        submission_config_payload = _legacy.load_config_payload(submission_config_path)
        Protenix = _legacy.import_protenix_class(CANDIDATE_DIR)
        if Protenix.__name__ != "Protenix":
            raise AssertionError(
                f"submission public class must be Protenix, got {Protenix.__name__}"
            )
        model, construction_metrics = _legacy.instantiate_submission_model(
            Protenix,
            submission_config_path,
            submission_config_payload,
            checkpoint_path,
        )
        _legacy.apply_evaluation_runtime_overrides(model)
        model.eval()
        update_check(
            current_index,
            True,
            f"Constructed candidate model through {construction_metrics['config_interface']}",
        )
        current_index += 1

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = model.to(device)
        update_check(current_index, True, f"Moved model to {device}")
        current_index += 1
        if not all(parameter.device == device for parameter in model.parameters()):
            raise AssertionError("Model parameters were not all moved to the selected device")
        update_check(current_index, True, "All model parameters are on the selected device")
        current_index += 1

        payload = torch.load(checkpoint_path, map_location=device)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("Official checkpoint is missing model state_dict")
        update_check(current_index, True, "Checkpoint payload contains model state_dict")
        current_index += 1
        state = payload["model"]
        if not isinstance(state, dict) or not state:
            raise TypeError("Official checkpoint model must be a nonempty state_dict")
        update_check(current_index, True, "Checkpoint model state is a nonempty mapping")
        current_index += 1
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        update_check(current_index, True, "Applied only the legacy module. key prefix normalization")
        current_index += 1

        target_state = model.state_dict()
        target_parameter_names = {name for name, _ in model.named_parameters()}
        official_names = set(state)
        mapping = _legacy._official_state_key_mapping(model)
        if official_names == target_parameter_names:
            key_mapping = {name: name for name in official_names}
            mapping_mode = "exact_parameter_keys"
        else:
            if _legacy._is_missing(mapping):
                raise AssertionError(
                    "Official checkpoint keys differ from candidate parameters without an "
                    "explicit OFFICIAL_STATE_DICT_KEY_MAP"
                )
            if not isinstance(mapping, dict):
                raise TypeError("Official state key mapping must be a dict")
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()):
                raise TypeError("Official state key mapping keys and values must be strings")
            key_mapping = dict(mapping)
            mapping_mode = "explicit_bijective_mapping"
            missing_official = sorted(official_names - set(key_mapping))
            unexpected_official = sorted(set(key_mapping) - official_names)
            mapped_targets = list(key_mapping.values())
            duplicate_targets = sorted(
                {name for name in mapped_targets if mapped_targets.count(name) > 1}
            )
            missing_targets = sorted(target_parameter_names - set(mapped_targets))
            unexpected_targets = sorted(set(mapped_targets) - target_parameter_names)
            if (
                missing_official
                or unexpected_official
                or duplicate_targets
                or missing_targets
                or unexpected_targets
            ):
                raise AssertionError(
                    "Official state key mapping is not a bijection over model parameters: "
                    f"missing_official={missing_official[:10]}, "
                    f"unexpected_official={unexpected_official[:10]}, "
                    f"duplicate_targets={duplicate_targets[:10]}, "
                    f"missing_targets={missing_targets[:10]}, "
                    f"unexpected_targets={unexpected_targets[:10]}"
                )
        update_check(current_index, True, f"Validated {mapping_mode}")
        current_index += 1

        remapped_state = {}
        for official_name, target_name in key_mapping.items():
            official_tensor = state[official_name]
            target_tensor = target_state[target_name]
            if tuple(official_tensor.shape) != tuple(target_tensor.shape):
                raise AssertionError(
                    f"Official weight shape mismatch: {official_name} -> {target_name}, "
                    f"official={tuple(official_tensor.shape)}, target={tuple(target_tensor.shape)}"
                )
            if official_tensor.dtype != target_tensor.dtype:
                raise AssertionError(
                    f"Official weight dtype mismatch: {official_name} -> {target_name}, "
                    f"official={official_tensor.dtype}, target={target_tensor.dtype}"
                )
            remapped_state[target_name] = official_tensor
        update_check(current_index, True, "Every mapped parameter has matching shape and dtype")
        current_index += 1

        preserved_buffer_names = sorted(set(target_state) - target_parameter_names)
        for buffer_name in preserved_buffer_names:
            remapped_state[buffer_name] = target_state[buffer_name]
        result = torch.nn.Module.load_state_dict(model, remapped_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise AssertionError(
                f"Official strict load is incomplete: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        update_check(
            current_index,
            True,
            f"Strictly loaded checkpoint with {len(preserved_buffer_names)} preserved buffers",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the legacy virtual-input and forward-output contracts."""
    CHECKPOINT_NAMES = [
        "candidate_model_for_forward",
        "official_weights_loaded_for_forward",
        "virtual_input_fixture",
        "input_required_fields",
        "input_tensor_shapes_and_devices",
        "input_atom_to_token_mapping",
        "input_atoms_per_token",
        "inference_forward_invocation",
        "forward_return_tuple",
        "prediction_dictionary",
        "contact_probability_contract",
        "required_prediction_fields",
        "finite_prediction_tensors",
        "coordinate_output_contract",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        submission_config_path = CANDIDATE_DIR / "inference_config.yaml"
        checkpoint_path = OFFICIAL_CHECKPOINT_PATH
        submission_config_payload = _legacy.load_config_payload(submission_config_path)
        Protenix = _legacy.import_protenix_class(CANDIDATE_DIR)
        if Protenix.__name__ != "Protenix":
            raise AssertionError(
                f"submission public class must be Protenix, got {Protenix.__name__}"
            )
        model, construction_metrics = _legacy.instantiate_submission_model(
            Protenix,
            submission_config_path,
            submission_config_payload,
            checkpoint_path,
        )
        _legacy.apply_evaluation_runtime_overrides(model)
        model.eval()
        update_check(current_index, True, f"Constructed model through {construction_metrics['config_interface']}")
        current_index += 1
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = model.to(device)
        _legacy.load_official_weights(torch, model, checkpoint_path, device)
        update_check(current_index, True, "Loaded official weights before inference")
        current_index += 1

        features = _legacy.make_virtual_protenix_input(torch, device)
        update_check(current_index, True, "Constructed the fixed 3-token, 9-atom virtual input")
        current_index += 1
        required = {
            "restype": (3, 32), "profile": (3, 32), "deletion_mean": (3, 1),
            "ref_pos": (9, 3), "ref_charge": (9,), "ref_mask": (9,),
            "ref_element": (9, 128), "ref_atom_name_chars": (9, 256),
            "ref_space_uid": (9,), "atom_to_token_idx": (9,),
            "atom_to_tokatom_idx": (9,), "distogram_rep_atom_mask": (9,),
            "asym_id": (3,), "residue_index": (3,), "entity_id": (3,),
            "token_index": (3,), "sym_id": (3,), "token_bonds": (3, 3),
            "has_frame": (3,), "is_ligand": (9,), "msa": (1, 3),
            "has_deletion": (1, 3), "deletion_value": (1, 3),
        }
        missing = sorted(set(required).difference(features))
        if missing:
            raise AssertionError(f"Preprocessed features are missing fields: {missing}")
        update_check(current_index, True, "All required virtual-input fields are present")
        current_index += 1
        for name, suffix in required.items():
            value = features[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Feature {name} is not a torch.Tensor")
            if tuple(value.shape) != suffix:
                raise AssertionError(
                    f"Feature {name} has shape {tuple(value.shape)}, expected {suffix}"
                )
            if value.device != features["restype"].device:
                raise AssertionError(f"Feature {name} is not on restype's device")
        update_check(current_index, True, "All feature tensors have required shapes and device")
        current_index += 1
        n_token = int(features["residue_index"].shape[-1])
        if not bool(((features["atom_to_token_idx"] >= 0) & (features["atom_to_token_idx"] < n_token)).all()):
            raise AssertionError("atom_to_token_idx is out of token range")
        if not bool(((features["atom_to_tokatom_idx"] >= 0) & (features["atom_to_tokatom_idx"] < 3)).all()):
            raise AssertionError("atom_to_tokatom_idx is out of virtual atom range")
        update_check(current_index, True, "Atom-to-token and atom-within-token indices are valid")
        current_index += 1
        atoms_per_token = torch.bincount(features["atom_to_token_idx"], minlength=n_token)
        expected_atoms_per_token = torch.full(
            (n_token,), 3, device=features["restype"].device, dtype=torch.long
        )
        if not bool(torch.equal(atoms_per_token, expected_atoms_per_token)):
            raise AssertionError("Each virtual residue must map to exactly three atoms")
        update_check(current_index, True, "Each virtual token has exactly three atoms")
        current_index += 1

        with torch.inference_mode():
            output = model(
                input_feature_dict=features,
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )
        update_check(current_index, True, "Inference forward executed under torch.inference_mode")
        current_index += 1
        if not isinstance(output, tuple) or len(output) != 3:
            raise AssertionError("Protenix.forward must return a three-item tuple")
        prediction, _, log = output
        update_check(current_index, True, "Forward returned (prediction, label, log)")
        current_index += 1
        if not isinstance(prediction, dict):
            raise AssertionError("prediction must be a dictionary")
        raw_prediction_keys = sorted(prediction)
        update_check(current_index, True, "Prediction is a dictionary")
        current_index += 1
        evaluation_prediction = dict(prediction)
        if "contact_probs" not in evaluation_prediction:
            full_data = prediction.get("full_data")
            if not isinstance(full_data, list) or not full_data:
                raise AssertionError("prediction lacks contact_probs and full_data is empty")
            first_sample = full_data[0]
            if not isinstance(first_sample, dict):
                raise AssertionError("prediction.full_data[0] must be a dictionary")
            contact_probs = first_sample.get("contact_probs")
            if not isinstance(contact_probs, torch.Tensor):
                raise AssertionError("prediction.full_data[0] lacks Tensor contact_probs")
            evaluation_prediction["contact_probs"] = contact_probs
        update_check(current_index, True, "contact_probs is available at top level or full_data[0]")
        current_index += 1
        missing_outputs = sorted(set(PREDICTION_FIELDS).difference(evaluation_prediction))
        if missing_outputs:
            raise AssertionError(f"prediction is missing fields: {missing_outputs}")
        update_check(current_index, True, "All required prediction fields are present")
        current_index += 1
        tensors = list(_legacy.tensor_values(evaluation_prediction))
        if not tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise AssertionError("prediction contains non-finite values")
        update_check(current_index, True, "All nested prediction tensors are finite")
        current_index += 1
        coordinate = evaluation_prediction["coordinate"]
        if not isinstance(coordinate, torch.Tensor) or coordinate.shape[-1] != 3:
            raise AssertionError(
                "coordinate must be a Tensor with final dimension 3, got "
                f"{getattr(coordinate, 'shape', None)}"
            )
        update_check(
            current_index,
            True,
            f"coordinate shape is {list(coordinate.shape)}; raw keys={raw_prediction_keys}; "
            f"log keys={sorted(log) if isinstance(log, dict) else []}",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the legacy deterministic and stochastic numerical oracle."""
    CHECKPOINT_NAMES = [
        "candidate_model_for_numerical_oracle",
        "official_weights_loaded_for_numerical_oracle",
        "reference_output_schema",
        "reference_deterministic_seed",
        "reference_stochastic_seeds",
        "deterministic_forward_run",
        "deterministic_replay",
        "deterministic_reference_comparison",
        "stochastic_forward_runs",
        "stochastic_reference_seed_101",
        "stochastic_reference_seed_202",
        "stochastic_reference_seed_303",
        "stochastic_distribution_comparison",
        "legacy_numerical_pass_condition",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        submission_config_path = CANDIDATE_DIR / "inference_config.yaml"
        checkpoint_path = OFFICIAL_CHECKPOINT_PATH
        submission_config_payload = _legacy.load_config_payload(submission_config_path)
        Protenix = _legacy.import_protenix_class(CANDIDATE_DIR)
        if Protenix.__name__ != "Protenix":
            raise AssertionError(
                f"submission public class must be Protenix, got {Protenix.__name__}"
            )
        model, construction_metrics = _legacy.instantiate_submission_model(
            Protenix,
            submission_config_path,
            submission_config_payload,
            checkpoint_path,
        )
        _legacy.apply_evaluation_runtime_overrides(model)
        model.eval()
        update_check(current_index, True, f"Constructed model through {construction_metrics['config_interface']}")
        current_index += 1
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = model.to(device)
        _legacy.load_official_weights(torch, model, checkpoint_path, device)
        update_check(current_index, True, "Loaded official weights before numerical inference")
        current_index += 1

        reference = _legacy.load_required_standard_output(OFFICIAL_REFERENCE_OUTPUT_PATH)
        deterministic_reference_section = reference["deterministic"]
        stochastic_reference_section = reference["stochastic"]
        update_check(current_index, True, "Loaded deterministic and stochastic reference sections")
        current_index += 1
        deterministic_seed = int(deterministic_reference_section.get("seed", -1))
        if deterministic_seed != DETERMINISTIC_SEED:
            raise AssertionError(
                f"Reference deterministic seed must be {DETERMINISTIC_SEED}, got {deterministic_seed}"
            )
        update_check(current_index, True, f"Reference deterministic seed is {DETERMINISTIC_SEED}")
        current_index += 1
        stochastic_seeds = tuple(stochastic_reference_section.get("seeds", ()))
        if stochastic_seeds != STOCHASTIC_SEEDS:
            raise AssertionError(
                f"Reference stochastic seeds must be {list(STOCHASTIC_SEEDS)}, "
                f"got {list(stochastic_seeds)}"
            )
        update_check(current_index, True, f"Reference stochastic seeds are {list(STOCHASTIC_SEEDS)}")
        current_index += 1

        _legacy.seed_random_sources(torch, DETERMINISTIC_SEED)
        deterministic_prediction, _, input_metrics, deterministic_output_metrics = (
            _legacy.run_forward_once(torch, model, device)
        )
        update_check(
            current_index,
            True,
            f"Ran deterministic inference with input={input_metrics} and output={deterministic_output_metrics}",
        )
        current_index += 1
        _legacy.seed_random_sources(torch, DETERMINISTIC_SEED)
        deterministic_replay_prediction, _, _, _ = _legacy.run_forward_once(
            torch, model, device
        )
        deterministic_replay_metrics = _legacy.compare_tensor_predictions(
            torch, deterministic_prediction, deterministic_replay_prediction
        )
        if not deterministic_replay_metrics["all_close"]:
            raise AssertionError("Deterministic replay does not satisfy legacy numerical tolerances")
        update_check(current_index, True, "Deterministic replay matches")
        current_index += 1

        deterministic_reference_prediction = _legacy.materialize_reference_prediction(
            torch, deterministic_reference_section
        )
        deterministic_reference_metrics = _legacy.compare_tensor_predictions(
            torch, deterministic_prediction, deterministic_reference_prediction
        )
        if not deterministic_reference_metrics["all_close"]:
            raise AssertionError("Deterministic prediction does not match the reference")
        update_check(current_index, True, "Deterministic prediction matches reference")
        current_index += 1

        stochastic_predictions = []
        stochastic_output_metrics = {}
        for seed in STOCHASTIC_SEEDS:
            _legacy.seed_random_sources(torch, seed)
            prediction, _, _, output_metrics = _legacy.run_forward_once(torch, model, device)
            stochastic_predictions.append(prediction)
            stochastic_output_metrics[str(seed)] = output_metrics
        update_check(current_index, True, "Ran stochastic inference for all legacy seeds")
        current_index += 1

        stochastic_reference_predictions = {}
        stochastic_per_seed_metrics = {}
        reference_prediction_payloads = stochastic_reference_section.get("predictions", {})
        for index, seed in enumerate(STOCHASTIC_SEEDS):
            raw_reference = reference_prediction_payloads.get(str(seed))
            if raw_reference is None:
                raise AssertionError(f"Reference stochastic.predictions lacks seed={seed}")
            reference_prediction = _legacy.materialize_reference_prediction(torch, raw_reference)
            stochastic_reference_predictions[str(seed)] = reference_prediction
            stochastic_per_seed_metrics[str(seed)] = _legacy.compare_tensor_predictions(
                torch, stochastic_predictions[index], reference_prediction
            )
            if not stochastic_per_seed_metrics[str(seed)]["all_close"]:
                raise AssertionError(f"Stochastic prediction for seed={seed} does not match reference")
            update_check(current_index, True, f"Stochastic prediction matches reference for seed={seed}")
            current_index += 1

        stochastic_actual_summary = _legacy.summarize_stochastic_predictions(
            torch, stochastic_predictions
        )
        stochastic_reference_summary = _legacy.summarize_stochastic_predictions(
            torch, [stochastic_reference_predictions[str(seed)] for seed in STOCHASTIC_SEEDS]
        )
        stochastic_distribution_metrics = _legacy.compare_distribution_summaries(
            stochastic_actual_summary, stochastic_reference_summary
        )
        if not stochastic_distribution_metrics["all_close"]:
            raise AssertionError("Stochastic prediction distribution does not match reference")
        update_check(current_index, True, "Stochastic distribution summary matches reference")
        current_index += 1

        numeric_passed = (
            deterministic_replay_metrics["all_close"]
            and deterministic_reference_metrics["all_close"]
            and all(metric["all_close"] for metric in stochastic_per_seed_metrics.values())
            and stochastic_distribution_metrics["all_close"]
        )
        if not numeric_passed:
            raise AssertionError("Legacy numerical pass condition is false")
        update_check(
            current_index,
            True,
            f"All legacy numerical comparisons passed; stochastic outputs={stochastic_output_metrics}",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
