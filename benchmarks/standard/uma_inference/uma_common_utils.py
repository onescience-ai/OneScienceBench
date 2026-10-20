
from __future__ import annotations

import datetime
import errno
import functools
import importlib
import logging
import os
import pathlib
import subprocess
import copy
from functools import reduce, wraps
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

import torch
import torch.nn as nn
import yaml

# Backward-compatible re-export; implementation lives in monitoring/uma.
from uma_runtime_logging import (
    SeverityLevelBetween,
    setup_logging,
)
from uma_registry import registry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.nn.modules.module import _IncompatibleKeys


DEFAULT_ENV_VARS = {
    # Expandable segments is a new cuda feature that helps with memory fragmentation during frequent
    # allocations (ie: in the case of variable batch sizes).
    # see https://pytorch.org/docs/stable/notes/cuda.html.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


# copied from https://stackoverflow.com/questions/33490870/parsing-yaml-in-python-detect-duplicated-keys
# prevents loading YAMLS where keys have been overwritten
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            each_key = self.construct_object(key_node, deep=deep)
            if each_key in mapping:
                raise ValueError(
                    f"Duplicate Key: {each_key!r} is found in YAML File.\n"
                    f"Error File location: {key_node.end_mark}"
                )
            mapping.add(each_key)
        return super().construct_mapping(node, deep)


def conditional_grad(dec):
    "Decorator to enable/disable grad depending on whether force/energy predictions are being made"

    # Adapted from https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


def _import_local_file(path: Path, *, project_root: Path) -> None:
    """
    Imports a Python file as a module

    :param path: The path to the file to import
    :type path: Path
    :param project_root: The root directory of the project (i.e., the "ocp" folder)
    :type project_root: Path
    """

    path = path.absolute()
    project_root = project_root.parent.absolute()

    module_name = ".".join(
        path.absolute().relative_to(project_root.absolute()).with_suffix("").parts
    )
    logging.debug(f"Resolved module name of {path} to {module_name}")
    importlib.import_module(module_name)


def setup_experimental_imports(project_root: Path) -> None:
    """
    Import selected directories of modules from the "experimental" subdirectory.

    If a file named ".include" is present in the "experimental" subdirectory,
    this will be read as a list of experimental subdirectories whose module
    (including in any subsubdirectories) should be imported.

    :param project_root: The root directory of the project (i.e., the "ocp" folder)
    """
    experimental_dir = (project_root / "experimental").absolute()
    if not experimental_dir.exists() or not experimental_dir.is_dir():
        return

    experimental_files = []
    include_file = experimental_dir / ".include"

    if include_file.exists():
        with open(include_file) as f:
            include_dirs = [line.rstrip("\n") for line in f.readlines() if line.strip()]

        for inc_dir in include_dirs:
            experimental_files.extend(
                f.absolute() for f in (experimental_dir / inc_dir).rglob("*.py")
            )

    for f in experimental_files:
        _import_local_file(f, project_root=project_root)


def _get_project_root() -> Path:
    """
    Gets the root folder of the project (the "ocp" folder)
    :return: The absolute path to the project root.
    """
    from uma_registry import registry

    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("onescience_models_UMA_root", no_warning=True)

    if root_folder is not None:
        assert isinstance(root_folder, str), "onescience_models_UMA_root must be a string"
        root_folder = Path(root_folder).resolve().absolute()
        assert root_folder.exists(), f"{root_folder} does not exist"
        assert root_folder.is_dir(), f"{root_folder} is not a directory"
    else:
        root_folder = Path(__file__).resolve().absolute().parent

    return root_folder


# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
def setup_imports(config: dict | None = None) -> None:
    from uma_registry import registry

    skip_experimental_imports = (config or {}).get("skip_experimental_imports", False)

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return

    try:
        project_root = _get_project_root()
        logging.info(f"Project root: {project_root}")
        importlib.import_module("uma_logger")

        import_keys = ["trainers", "datasets", "models", "tasks"]
        for key in import_keys:
            for f in (project_root / "core" / key).rglob("*.py"):
                _import_local_file(f, project_root=project_root)

        if not skip_experimental_imports:
            setup_experimental_imports(project_root)
    finally:
        registry.register("imports_setup", True)


def debug_log_entry_exit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"{func.__name__}...")
        result = func(*args, **kwargs)
        logging.debug(f"{func.__name__} done")
        return result

    return wrapper


def setup_env_vars() -> None:
    for k, v in DEFAULT_ENV_VARS.items():
        os.environ[k] = v
        logging.info(f"Setting env {k}={v}")


def _resolve_scale_factor_submodule(model: nn.Module, name: str):
    from uma_scale_factor import ScaleFactor

    try:
        scale = model.get_submodule(name)
        if not isinstance(scale, ScaleFactor):
            return None
        return scale
    except AttributeError:
        return None


def _report_incompat_keys(
    model: nn.Module,
    keys: _IncompatibleKeys,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    # filter out the missing scale factor keys for the new scaling factor module
    missing_keys: list[str] = []
    for full_key_name in keys.missing_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(model, parent_module_name)
        if scale_factor is not None:
            continue
        missing_keys.append(full_key_name)

    # filter out unexpected scale factor keys that remain from the old scaling modules
    unexpected_keys: list[str] = []
    for full_key_name in keys.unexpected_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(model, parent_module_name)
        if scale_factor is not None:
            continue
        unexpected_keys.append(full_key_name)

    error_msgs = []
    if len(unexpected_keys) > 0:
        error_msgs.insert(
            0,
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join(f'"{k}"' for k in unexpected_keys)
            ),
        )
    if len(missing_keys) > 0:
        error_msgs.insert(
            0,
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join(f'"{k}"' for k in missing_keys)
            ),
        )

    if len(error_msgs) > 0:
        error_msg = "Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)
        )
        if strict:
            raise RuntimeError(error_msg)
        logging.warning(error_msg)

    return missing_keys, unexpected_keys


def match_state_dict(
    model_state_dict: Mapping[str, torch.Tensor],
    checkpoint_state_dict: Mapping[str, torch.Tensor],
) -> dict:
    # match the model's state dict with the checkpoint state and return a new dict
    # that's compatible with the models

    # Match the "module." count in the keys of model and checkpoint state_dict
    # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
    # Not using either of the above two would have no "module."

    ckpt_key_count = next(iter(checkpoint_state_dict)).count("module")
    mod_key_count = next(iter(model_state_dict)).count("module")
    key_count_diff = mod_key_count - ckpt_key_count

    if key_count_diff > 0:
        new_dict = {
            key_count_diff * "module." + k: v for k, v in checkpoint_state_dict.items()
        }
    elif key_count_diff < 0:
        new_dict = {
            k[len("module.") * abs(key_count_diff) :]: v
            for k, v in checkpoint_state_dict.items()
        }
    else:
        new_dict = checkpoint_state_dict
    return new_dict


def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    incompat_keys = module.load_state_dict(state_dict, strict=False)  # type: ignore
    return _report_incompat_keys(module, incompat_keys, strict=strict)



def get_commit_hash() -> str:
    core_hash = get_commit_hash_for_repo(str(Path(__file__).resolve().parent))
    experimental_hash = None
    return f"UMA:{core_hash or 'NA'},experimental:{experimental_hash or 'NA'}"




def get_commit_hash_for_repo(
    git_repo_path: str,
) -> str | None:
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "-C", git_repo_path, "describe", "--always"],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode("ascii")
        )
    # catch instances where code is not being run from a git repo
    except Exception:
        commit_hash = None

    return commit_hash


def load_model_and_weights_from_checkpoint(checkpoint_path: str) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            errno.ENOENT, "Checkpoint file not found", checkpoint_path
        )
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )

    def rewrite_checkpoint_symbol(value: str) -> str:
        exact_map = {
            "onescience.models.UMA.base.HydraModel": "hydra",
            "onescience.models.UMA.base.HydraModelV2": "HydraModelV2",
            "onescience.models.UMA.uma_escn_md.eSCNMDBackbone": "escnmd_backbone",
            "onescience.models.UMA.uma_escn_moe.eSCNMDMoeBackbone": "escnmd_moe_backbone",
            "onescience.models.UMA.uma_escn_moe.DatasetSpecificSingleHeadWrapper": "DatasetSpecificSingleHeadWrapper",
            "onescience.models.UMA.uma_escn_moe.DatasetSpecificMoEWrapper": "DatasetSpecificMoEWrapper",
            "onescience.models.UMA.uma_escn_md.MLP_EFS_Head": "MLP_EFS_Head",
            "onescience.models.UMA.uma_escn_md.MLP_Energy_Head": "MLP_Energy_Head",
            "onescience.models.UMA.uma_escn_md.MLP_Stress_Head": "MLP_Stress_Head",
            "onescience.models.UMA.uma_escn_md.Linear_Energy_Head": "Linear_Energy_Head",
            "onescience.models.UMA.uma_escn_md.Linear_Force_Head": "Linear_Force_Head",
            "onescience.models.UMA.models.base.HydraModel": "hydra",
            "onescience.models.UMA.models.base.HydraModelV2": "HydraModelV2",
            "onescience.models.UMA.models.uma.escn_md.eSCNMDBackbone": "escnmd_backbone",
            "onescience.models.UMA.models.uma.escn_moe.eSCNMDMoeBackbone": "escnmd_moe_backbone",
            "onescience.models.UMA.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper": "DatasetSpecificSingleHeadWrapper",
            "onescience.models.UMA.models.uma.escn_moe.DatasetSpecificMoEWrapper": "DatasetSpecificMoEWrapper",
            "onescience.models.UMA.models.uma.escn_md.MLP_EFS_Head": "MLP_EFS_Head",
            "onescience.models.UMA.models.uma.escn_md.MLP_Energy_Head": "MLP_Energy_Head",
            "onescience.models.UMA.models.uma.escn_md.MLP_Stress_Head": "MLP_Stress_Head",
            "onescience.models.UMA.models.uma.escn_md.Linear_Energy_Head": "Linear_Energy_Head",
            "onescience.models.UMA.models.uma.escn_md.Linear_Force_Head": "Linear_Force_Head",
        }
        if value in exact_map:
            return exact_map[value]
        return value

    def normalize_checkpoint_config(payload):
        if isinstance(payload, Mapping):
            normalized = {}
            for key, value in dict(payload).items():
                if key in {"_target_", "_recursive_", "_convert_", "_partial_"}:
                    continue
                if key in {"model", "module", "head_cls", "name"} and isinstance(value, str):
                    normalized[key] = rewrite_checkpoint_symbol(value)
                else:
                    normalized[key] = normalize_checkpoint_config(value)
            return normalized
        if isinstance(payload, list):
            return [normalize_checkpoint_config(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(normalize_checkpoint_config(item) for item in payload)
        if hasattr(payload, "items") and callable(getattr(payload, "items")):
            normalized = {}
            for key, value in dict(payload.items()).items():
                if key in {"_target_", "_recursive_", "_convert_", "_partial_"}:
                    continue
                if key in {"model", "module", "head_cls", "name"} and isinstance(value, str):
                    normalized[key] = rewrite_checkpoint_symbol(value)
                else:
                    normalized[key] = normalize_checkpoint_config(value)
            return normalized
        return payload

    def extract_checkpoint_payload(payload):
        visited = set()

        def walk(obj):
            obj_id = id(obj)
            if obj_id in visited:
                return None
            visited.add(obj_id)

            if isinstance(obj, dict):
                if "config" in obj and "state_dict" in obj:
                    config_obj = obj["config"]
                    if isinstance(config_obj, dict) and "model" in config_obj:
                        return {
                            "config": copy.deepcopy(config_obj["model"]),
                            "state_dict": obj["state_dict"],
                        }
                if "model_config" in obj and "model_state_dict" in obj:
                    return {
                        "config": copy.deepcopy(obj["model_config"]),
                        "state_dict": obj["model_state_dict"],
                    }
                for value in obj.values():
                    found = walk(value)
                    if found is not None:
                        return found

            # Some checkpoints may deserialize into custom wrapper objects.
            for config_attr, state_attr in (
                ("model_config", "model_state_dict"),
                ("config", "state_dict"),
            ):
                if hasattr(obj, config_attr) and hasattr(obj, state_attr):
                    config_obj = getattr(obj, config_attr)
                    state_obj = getattr(obj, state_attr)
                    if config_attr == "config" and isinstance(config_obj, dict) and "model" in config_obj:
                        config_obj = config_obj["model"]
                    return {
                        "config": copy.deepcopy(config_obj),
                        "state_dict": state_obj,
                    }

            if isinstance(obj, (list, tuple)):
                for item in obj:
                    found = walk(item)
                    if found is not None:
                        return found
            return None

        return walk(payload)

    extracted = extract_checkpoint_payload(checkpoint)
    if extracted is None:
        details = type(checkpoint).__name__
        if isinstance(checkpoint, dict):
            details = f"dict keys={sorted(checkpoint.keys())}"
        raise KeyError(
            "Unsupported UMA checkpoint format. Expected either "
            "`config`+`state_dict` or `model_config`+`model_state_dict`. "
            f"Observed top-level payload: {details}"
        )

    config = normalize_checkpoint_config(extracted["config"])
    if "name" in config:
        name = rewrite_checkpoint_symbol(config.pop("name"))
    elif "model" in config and isinstance(config["model"], str):
        name = rewrite_checkpoint_symbol(config.pop("model"))
    else:
        name = "hydra"
    state_dict = extracted["state_dict"]

    model = registry.get_model_class(name)(**config)
    matched_dict = match_state_dict(model.state_dict(), state_dict)
    load_state_dict(model, matched_dict, strict=True)
    return model


def get_timestamp_uid() -> str:
    return datetime.datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())[:4]


@torch.no_grad()
def tensor_stats(name: str, x: torch.Tensor) -> dict:
    return {
        f"{name}.max": x.max().item(),
        f"{name}.min": x.min().item(),
        f"{name}.std": x.std().item(),
        f"{name}.mean": x.mean().item(),
        f"{name}.norm": torch.norm(x, p=2).item(),
        f"{name}.nonzero_fraction": torch.nonzero(x).shape[0] / float(x.numel()),
    }


def get_weight_table(model: torch.nn.Module) -> tuple[list, list]:
    stat_names = list(tensor_stats("weight", torch.Tensor([1])).keys())
    columns = ["ParamName", "shape"] + stat_names + ["grad." + n for n in stat_names]
    data = []
    for param_name, params in model.named_parameters():
        row_weight = list(tensor_stats(f"weights/{param_name}", params).values())
        if params.grad is not None:
            row_grad = list(tensor_stats(f"grad/{param_name}", params.grad).values())
        else:
            row_grad = [None] * len(row_weight)
        data.append([param_name] + [params.shape] + row_weight + row_grad)
    return columns, data


def get_checkpoint_format(config: dict) -> str:
    # a temporary function to retrieve the checkpoint format from old configs
    format = config.get("optim", {}).get("checkpoint_format", "pt")
    assert format in (
        "pt",
        "dcp",
    ), f"checkpoint format can only be pt or dcp, found {format}"
    return format


def get_deep(dictionary: dict, keys: str, default: str | None = None):
    # given a nested dictionary and a dot separated query, retrieve the item
    # example:
    # get_deep(dictionary{"oc20":{"energy",1}}, keys="oc20.energy") -> 1
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )


def get_subdirectories_sorted_by_time(directory: str) -> list:
    """
    Get all subdirectories in a directory sorted by their last modification time.
    Args:
        directory (str): The path to the directory to search.
    Returns:
        list: A list of tuples containing the subdirectory path and its last modification time.
    """
    if not os.path.exists(directory):
        return []

    directory = pathlib.Path(directory)
    return sorted(
        ((str(d), d.stat().st_mtime) for d in directory.iterdir() if d.is_dir()),
        key=lambda x: x[1],
    )
