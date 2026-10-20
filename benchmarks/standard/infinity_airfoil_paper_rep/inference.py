"""Inference for the flattened 2-D INFINITY AirfRANS model.

At test time only geometry is observed. The script:
1. auto-decodes SDF and wall normals in exactly three configured steps;
2. feeds [z_d, z_n, Vx, Vy] through g_psi;
3. decodes vx, vy, p and nu_t on the requested mesh;
4. writes physical-unit predictions and normalized MSE diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from datset import (
    AirfRANSDataset,
    NormalizationStats,
    field_training_tensors,
    full_decode_coordinates,
    load_full_split,
)
from model import GEOMETRY_FIELDS, PHYSICS_FIELDS, adapt_code, build_inr, build_processor
from utils import decode_in_chunks, freeze_module, move_code_stats, resolve_device


def load_bundle(path: str | Path, device: torch.device):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    stats = NormalizationStats.from_dict(bundle["data_stats"])

    inrs = {}
    for field, state in bundle["inrs"].items():
        model = build_inr(field, config).to(device)
        model.load_state_dict(state)
        inrs[field] = freeze_module(model)

    processor = build_processor(config).to(device)
    processor.load_state_dict(bundle["processor"])
    freeze_module(processor)
    code_stats = move_code_stats(bundle["code_stats"], device)
    return config, stats, inrs, processor, code_stats


def encode_geometry(
    case,
    stats: NormalizationStats,
    inrs: Mapping,
    config: Mapping,
    code_stats: Mapping,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    steps = int(config["optimization"]["test_inner_steps"])
    n_points = int(config["data"]["sample_points_per_case"])
    use_sdf = bool(config["model"].get("physics_decoder_condition_on_sdf", False))
    codes = []

    # Fixed seed makes any 4k mesh subsampling deterministic.
    generator = torch.Generator().manual_seed(int(config["runtime"]["seed"]))
    for field in GEOMETRY_FIELDS:
        coordinates, target = field_training_tensors(
            case,
            stats,
            field,
            n_points=n_points,
            physics_decoder_condition_on_sdf=use_sdf,
            generator=generator,
        )
        coordinates = coordinates.to(device)
        target = target.to(device)
        with torch.enable_grad():
            code = adapt_code(
                inrs[field], coordinates, target, steps, create_graph=False
            ).detach()
        normalized = (code - code_stats[field]["mean"]) / code_stats[field]["std"]
        codes.append(normalized)
    return codes[0], codes[1]



def predict_case(
    case,
    stats: NormalizationStats,
    inrs: Mapping,
    processor,
    code_stats: Mapping,
    config: Mapping,
    device: torch.device,
    *,
    chunk_size: int = 100_000,
) -> tuple[Tensor, dict[str, float]]:
    z_sdf, z_n = encode_geometry(case, stats, inrs, config, code_stats, device)
    inlet = stats.normalized_inlet(case).to(device)
    processor_input = torch.cat((z_sdf, z_n, inlet), dim=-1).unsqueeze(0)

    with torch.no_grad():
        predicted_normalized_codes = processor.split_output(processor(processor_input))

    use_sdf = bool(config["model"].get("physics_decoder_condition_on_sdf", False))
    coordinates = full_decode_coordinates(
        case, stats, physics_decoder_condition_on_sdf=use_sdf
    ).to(device)

    normalized_fields = []
    for field in PHYSICS_FIELDS:
        z_hat_norm = predicted_normalized_codes[field][0]
        z_hat = z_hat_norm * code_stats[field]["std"] + code_stats[field]["mean"]
        normalized_fields.append(
            decode_in_chunks(inrs[field], coordinates, z_hat, chunk_size)
        )
    normalized_prediction = torch.cat(normalized_fields, dim=-1)
    physical_prediction = stats.denormalize_fields(normalized_prediction)

    truth_norm = stats.normalized_fields(case)
    metrics = {
        "mse_vx_normalized": float(F.mse_loss(normalized_prediction[:, 0], truth_norm[:, 0])),
        "mse_vy_normalized": float(F.mse_loss(normalized_prediction[:, 1], truth_norm[:, 1])),
        "mse_p_normalized": float(F.mse_loss(normalized_prediction[:, 2], truth_norm[:, 2])),
        "mse_nu_normalized": float(F.mse_loss(normalized_prediction[:, 3], truth_norm[:, 3])),
    }
    if bool(case.surface.any()):
        metrics["mse_surface_p_normalized"] = float(
            F.mse_loss(
                normalized_prediction[case.surface, 2],
                truth_norm[case.surface, 2],
            )
        )
    return physical_prediction, metrics


def save_npz(path: str | Path, case, prediction: Tensor, metrics: Mapping[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = prediction.numpy()
    np.savez_compressed(
        path,
        case=np.asarray(case.name),
        xy=case.xy.numpy(),
        surface=case.surface.numpy(),
        pred_vx=pred[:, 0],
        pred_vy=pred[:, 1],
        pred_p=pred[:, 2],
        pred_nu_t=pred[:, 3],
        true_vx=case.fields[:, 0].numpy(),
        true_vy=case.fields[:, 1].numpy(),
        true_p=case.fields[:, 2].numpy(),
        true_nu_t=case.fields[:, 3].numpy(),
        **{key: np.asarray(value) for key, value in metrics.items()},
    )


def save_vtk(path: str | Path, dataset: AirfRANSDataset, case, prediction: Tensor) -> None:
    """Attach predictions to the original internal mesh for downstream force evaluation."""
    import pyvista as pv

    internal_path = dataset.dataset_dir / case.name / f"{case.name}_internal.vtu"
    mesh = pv.read(internal_path)
    pred = prediction.numpy()
    mesh.point_data["pred_U"] = np.column_stack((pred[:, :2], np.zeros(len(pred), dtype=pred.dtype)))
    mesh.point_data["pred_p"] = pred[:, 2]
    mesh.point_data["pred_nut"] = pred[:, 3]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/infinity_airfoil.pt")
    parser.add_argument("--data-root", default=None, help="Override data.root saved in the checkpoint")
    parser.add_argument("--case", default=None, help="AirfRANS case name; defaults to --split/--index")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--output", default="prediction.npz")
    parser.add_argument("--write-vtk", default=None, metavar="PATH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config, stats, inrs, processor, code_stats = load_bundle(args.checkpoint, device)
    if args.data_root:
        config["data"]["root"] = args.data_root

    trainset, testset = load_full_split(config)
    dataset = trainset if args.split == "train" else testset
    if args.case is not None:
        try:
            index = dataset.names.index(args.case)
        except ValueError:
            # Permit any AirfRANS case folder under Dataset, including external cases.
            dataset = AirfRANSDataset(config["data"]["root"], [args.case], cache_dir=config["data"].get("cache_dir"))
            index = 0
    else:
        index = args.index

    case = dataset[index]
    prediction, metrics = predict_case(
        case,
        stats,
        inrs,
        processor,
        code_stats,
        config,
        device,
        chunk_size=args.chunk_size,
    )
    save_npz(args.output, case, prediction, metrics)
    if args.write_vtk:
        save_vtk(args.write_vtk, dataset, case, prediction)

    print(f"case={case.name}")
    for key, value in metrics.items():
        print(f"{key}={value:.6e}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
