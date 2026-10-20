from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path_value):
    """Resolve a project-relative path without depending on the current directory."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path_value="config/config.yaml"):
    """Load the project YAML file as a plain dictionary."""
    config_path = resolve_path(path_value)
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config, config_path
