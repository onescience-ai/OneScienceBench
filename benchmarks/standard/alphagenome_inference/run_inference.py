"""Standalone lightweight AlphaGenome-style model and inference entry point."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DNA_ALPHABET = "ACGT"
TRACK_CHANNELS = {
    "atac": 3,
    "dnase": 2,
    "cage": 4,
    "rna_seq": 2,
    "chip_tf": 5,
    "chip_histone": 3,
}


class ResidualConvBlock(nn.Module):
    """Two same-width convolutions with normalization and a residual path."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, inputs):
        hidden = F.gelu(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        return F.gelu(inputs + hidden)


class AlphaGenomeTransformerBlock(nn.Module):
    """Pre-normalized self-attention and feed-forward residual block."""

    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, inputs):
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden = inputs + attended
        return hidden + self.ffn(self.ffn_norm(hidden))


class AlphaGenomeModel(nn.Module):
    """Compact sequence-to-track architecture inspired by AlphaGenome."""

    def __init__(
        self,
        hidden_dim=16,
        num_heads=4,
        num_transformer_layers=2,
        track_channels=None,
    ):
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4 for GroupNorm")
        if (hidden_dim * 2) % num_heads != 0:
            raise ValueError("2 * hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.track_channels = dict(track_channels or TRACK_CHANNELS)

        self.stem = nn.Conv1d(4, hidden_dim, kernel_size=7, padding=3)
        self.encoder = ResidualConvBlock(hidden_dim)
        self.downsample = nn.Conv1d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.transformer = nn.ModuleList(
            [
                AlphaGenomeTransformerBlock(
                    hidden_dim=hidden_dim * 2,
                    num_heads=num_heads,
                    ffn_dim=hidden_dim * 4,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.bottleneck_norm = nn.LayerNorm(hidden_dim * 2)
        self.upsample_projection = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.decoder = ResidualConvBlock(hidden_dim)
        self.output_heads = nn.ModuleDict(
            {
                name: nn.Conv1d(hidden_dim, channels, kernel_size=1)
                for name, channels in self.track_channels.items()
            }
        )

    def forward(self, dna_one_hot):
        if dna_one_hot.ndim != 3 or dna_one_hot.shape[-1] != 4:
            raise ValueError("dna_one_hot must have shape [batch, length, 4]")
        if dna_one_hot.shape[1] < 4:
            raise ValueError("DNA sequence length must be at least 4")

        hidden = dna_one_hot.transpose(1, 2)
        skip = self.encoder(self.stem(hidden))
        hidden = self.downsample(skip).transpose(1, 2)
        for block in self.transformer:
            hidden = block(hidden)
        hidden = self.bottleneck_norm(hidden).transpose(1, 2)
        hidden = F.interpolate(
            hidden,
            size=skip.shape[-1],
            mode="linear",
            align_corners=False,
        )
        hidden = self.upsample_projection(hidden) + skip
        hidden = self.decoder(hidden)
        return {
            name: F.softplus(head(hidden).transpose(1, 2))
            for name, head in self.output_heads.items()
        }


def encode_dna(sequences):
    if isinstance(sequences, str):
        sequences = [sequences]
    if not sequences:
        raise ValueError("At least one DNA sequence is required")
    normalized = [sequence.strip().upper() for sequence in sequences]
    sequence_length = len(normalized[0])
    if sequence_length < 4:
        raise ValueError("DNA sequence length must be at least 4")
    if any(len(sequence) != sequence_length for sequence in normalized):
        raise ValueError("All DNA sequences in a batch must have equal length")

    encoded = torch.zeros(len(normalized), sequence_length, 4, dtype=torch.float32)
    base_to_index = {base: index for index, base in enumerate(DNA_ALPHABET)}
    for batch_index, sequence in enumerate(normalized):
        invalid = sorted(set(sequence) - set(DNA_ALPHABET))
        if invalid:
            raise ValueError(f"Unsupported DNA bases: {invalid}")
        for position, base in enumerate(sequence):
            encoded[batch_index, position, base_to_index[base]] = 1.0
    return encoded


def load_checkpoint(model, checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)


def save_predictions(predictions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for track_name in TRACK_CHANNELS:
        values = predictions[track_name][0].detach().cpu().numpy()
        output_path = output_dir / f"{track_name}.npy"
        np.save(output_path, values)
        saved_paths.append(output_path)
    return saved_paths


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but no accelerator is available")
    return torch.device(requested)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run a standalone lightweight AlphaGenome-style model"
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main():
    args = create_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = resolve_device(args.device)
    model = AlphaGenomeModel().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    dna_one_hot = encode_dna(args.sequence).to(device)
    with torch.no_grad():
        predictions = model(dna_one_hot)
    saved_paths = save_predictions(predictions, args.output_dir)
    if len(saved_paths) != len(TRACK_CHANNELS):
        raise RuntimeError("Not all AlphaGenome tracks were saved")


if __name__ == "__main__":
    main()
