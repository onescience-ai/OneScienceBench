from unetex import UNetEx


def build_model(config):
    """Create the single model used by this project."""
    config = dict(config)
    config.pop("name", None)  # Keep existing checkpoints backward compatible.
    return UNetEx(**config)
