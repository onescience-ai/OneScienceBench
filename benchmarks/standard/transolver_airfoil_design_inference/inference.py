"""Run standalone Transolver inference on the AirfRANS test set."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from AirfRANSDataSet import AirfRANSDatapipe
from Transolver2D import Transolver2D
from metrics import AirfRANSEvaluator, json_ready, seed_everything


MODEL_NAME = "Transolver"
TASK_NAME = "full"
SUBSAMPLING = 32_000
SEED = 42
DEFAULT_DATA_DIR = Path(
    "/public/share/sugonhpcapp01/onestore/onedatasets/"
    "Transolver-Airfoil-Design/Dataset"
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:0")
    parser.add_argument("--checkpoint", default=ROOT / "weight" / "best.pth")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", default=ROOT / "result")
    parser.add_argument(
        "--n-test", type=int, default=3, help="number of VTK results to save"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_data_config(data_dir: Path, num_workers: int) -> SimpleNamespace:
    """Build the small attribute-based configuration expected by the dataset."""
    return SimpleNamespace(
        verbose=False,
        source=SimpleNamespace(data_dir=str(data_dir)),
        data=SimpleNamespace(
            splits=SimpleNamespace(
                train_name="full_train",
                test_name="full_test",
                val_split_ratio=0.1,
            ),
            sampling=SimpleNamespace(sample_strategy=None),
            crop=None,
            subsampling=SUBSAMPLING,
        ),
        dataloader=SimpleNamespace(batch_size=1, num_workers=num_workers),
    )


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        return torch.device(requested if torch.cuda.is_available() else "cpu")
    if requested != "auto":
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise TypeError(f"Checkpoint has no model_state_dict: {path}")
    return checkpoint


def run(args: argparse.Namespace, logger: logging.Logger) -> int:
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")

    data_dir = resolve_path(args.data_dir)
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.result_dir) / TASK_NAME
    device = select_device(args.device)
    logger.info("Device: %s", device)

    datapipe = AirfRANSDatapipe(
        build_data_config(data_dir, args.num_workers), distributed=False
    )
    test_loader = datapipe.test_dataloader()
    test_names = list(datapipe.test_dataset.data_list_names)

    # Keep the checkpoint-compatible model parameters next to the inference entrypoint.
    model = Transolver2D(
        n_hidden=256,
        n_layers=8,
        space_dim=7,
        fun_dim=0,
        n_head=8,
        mlp_ratio=2,
        out_dim=4,
        slice_num=32,
        unified_pos=True,
    ).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    evaluator = AirfRANSEvaluator(
        device,
        models=[model],
        hparams=[{"subsampling": SUBSAMPLING}],
        coef_norm=datapipe.coef_norm,
    )
    sample_count = len(test_names)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)
    save_count = min(max(args.n_test, 0), sample_count)
    save_indices = sorted(random.sample(range(sample_count), save_count))

    score = evaluator.evaluate(
        test_loader,
        test_names,
        data_dir,
        max_samples=sample_count,
        save_indices=save_indices,
        output_dir=output_dir,
        model_name=MODEL_NAME,
        progress=True,
    )
    score_path = output_dir / f"score_{MODEL_NAME}.json"
    score_path.write_text(
        json.dumps(json_ready(score), indent=2), encoding="utf-8"
    )

    metrics = evaluator.paper_metrics(score)
    logger.info("Checkpoint: %s (epoch=%s)", checkpoint_path, checkpoint.get("epoch"))
    logger.info("Saved AirfRANS outputs to %s", output_dir)
    logger.info("Volume: %.10f", metrics["Volume"])
    logger.info(
        "Surf: %.10f; pressure term: %.10f",
        metrics["Surf"],
        metrics["Surf_pressure"],
    )
    logger.info("CL relative error: %.10f", metrics["CL_relative_error"])
    logger.info("rho_L: %.10f", metrics["rho_L"])
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    seed_everything(SEED)
    return run(args, logging.getLogger("transolver-inference"))


if __name__ == "__main__":
    raise SystemExit(main())
