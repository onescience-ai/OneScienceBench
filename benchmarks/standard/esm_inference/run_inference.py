"""Standalone lightweight ESM-style model and inference workflows."""

import argparse
import csv
from pathlib import Path
import re

import torch
from torch import nn


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>")
VOCABULARY = SPECIAL_TOKENS + tuple(AMINO_ACIDS)


class ProteinTokenizer:
    """Fixed ESM-style protein tokenizer with CLS, EOS, and padding tokens."""

    def __init__(self):
        self.token_to_index = {
            token: index for index, token in enumerate(VOCABULARY)
        }
        self.pad_index = self.token_to_index["<pad>"]
        self.cls_index = self.token_to_index["<cls>"]
        self.eos_index = self.token_to_index["<eos>"]
        self.unknown_index = self.token_to_index["<unk>"]

    def encode(self, sequences):
        if isinstance(sequences, str):
            sequences = [sequences]
        if not sequences:
            raise ValueError("At least one protein sequence is required")
        normalized = [sequence.strip().upper() for sequence in sequences]
        if any(not sequence for sequence in normalized):
            raise ValueError("Protein sequences must not be empty")
        invalid = sorted(set("".join(normalized)) - set(AMINO_ACIDS))
        if invalid:
            raise ValueError(f"Unsupported amino acids: {invalid}")

        max_tokens = max(len(sequence) for sequence in normalized) + 2
        token_ids = torch.full(
            (len(normalized), max_tokens),
            self.pad_index,
            dtype=torch.long,
        )
        for row, sequence in enumerate(normalized):
            token_ids[row, 0] = self.cls_index
            for column, residue in enumerate(sequence, start=1):
                token_ids[row, column] = self.token_to_index[residue]
            token_ids[row, len(sequence) + 1] = self.eos_index
        return token_ids

    def get_idx(self, residue):
        if residue not in AMINO_ACIDS:
            raise ValueError(f"Unsupported amino acid: {residue}")
        return self.token_to_index[residue]


class ESMTransformerLayer(nn.Module):
    """Pre-normalized self-attention and feed-forward ESM block."""

    def __init__(self, embedding_dim, num_heads, ffn_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
        )

    def forward(self, inputs, padding_mask):
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = inputs + attended
        return hidden + self.ffn(self.ffn_norm(hidden))


class ESMBackbone(nn.Module):
    """Token/position embeddings followed by a fixed Transformer stack."""

    def __init__(
        self,
        vocab_size=len(VOCABULARY),
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        max_length=128,
        padding_index=0,
    ):
        super().__init__()
        self.max_length = max_length
        self.padding_index = padding_index
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_index,
        )
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.layers = nn.ModuleList(
            [
                ESMTransformerLayer(embedding_dim, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if token_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Token length {token_ids.shape[1]} exceeds max_length {self.max_length}"
            )
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[
            None, :, :
        ]
        padding_mask = token_ids.eq(self.padding_index)
        for layer in self.layers:
            hidden = layer(hidden, padding_mask)
        return self.final_norm(hidden)


class ESMInferenceModel(nn.Module):
    """Compact ESM backbone with language-model and coordinate heads."""

    def __init__(
        self,
        vocab_size=len(VOCABULARY),
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        max_length=128,
        padding_index=0,
    ):
        super().__init__()
        self.backbone = ESMBackbone(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_length=max_length,
            padding_index=padding_index,
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.coordinate_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 3),
        )

    def forward(self, token_ids):
        representations = self.backbone(token_ids)
        return {
            "representations": representations,
            "logits": self.lm_head(representations),
            "coordinates": self.coordinate_head(representations),
        }


def read_fasta(path):
    records = []
    label = None
    sequence_parts = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    records.append((label, "".join(sequence_parts)))
                label = line[1:].strip()
                sequence_parts = []
            else:
                if label is None:
                    raise ValueError("FASTA sequence found before a header")
                sequence_parts.append(line)
    if label is not None:
        records.append((label, "".join(sequence_parts)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def safe_output_name(label):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not name:
        raise ValueError(f"FASTA label cannot be used as a file name: {label!r}")
    return name


def load_checkpoint(model, checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)


def extract_representations(model, tokenizer, records, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, sequence in records:
        token_ids = tokenizer.encode(sequence).to(device)
        with torch.no_grad():
            outputs = model(token_ids)
        per_token = outputs["representations"][0, 1 : len(sequence) + 1].cpu()
        result = {
            "label": label,
            "representations": per_token,
            "mean_representations": per_token.mean(dim=0),
        }
        torch.save(result, output_dir / f"{safe_output_name(label)}.pt")


def format_pdb(sequence, coordinates):
    lines = ["HEADER    LIGHTWEIGHT ESM STRUCTURE"]
    for index, (residue, coordinate) in enumerate(zip(sequence, coordinates), start=1):
        x_value, y_value, z_value = [float(value) for value in coordinate]
        lines.append(
            f"ATOM  {index:5d}  CA  {residue:>3s} A{index:4d}    "
            f"{x_value:8.3f}{y_value:8.3f}{z_value:8.3f}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def fold_structures(model, tokenizer, records, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, sequence in records:
        token_ids = tokenizer.encode(sequence).to(device)
        with torch.no_grad():
            outputs = model(token_ids)
        coordinates = outputs["coordinates"][0, 1 : len(sequence) + 1].cpu()
        pdb_text = format_pdb(sequence, coordinates)
        (output_dir / f"{safe_output_name(label)}.pdb").write_text(
            pdb_text,
            encoding="utf-8",
        )


def parse_mutation(mutation, sequence, offset_idx):
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation)
    if match is None:
        raise ValueError(f"Invalid single-site mutation: {mutation!r}")
    wildtype, position_text, mutant = match.groups()
    sequence_index = int(position_text) - offset_idx
    if sequence_index < 0 or sequence_index >= len(sequence):
        raise ValueError(f"Mutation position is outside the sequence: {mutation}")
    if sequence[sequence_index] != wildtype:
        raise ValueError(
            f"Wildtype mismatch for {mutation}: expected {sequence[sequence_index]}"
        )
    return wildtype, sequence_index, mutant


def score_variants(
    model,
    tokenizer,
    sequence,
    dms_input,
    mutation_column,
    offset_idx,
    output_path,
    device,
):
    with dms_input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or mutation_column not in reader.fieldnames:
            raise ValueError(f"Missing mutation column: {mutation_column}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError(f"No variants found in {dms_input}")

    token_ids = tokenizer.encode(sequence).to(device)
    with torch.no_grad():
        log_probabilities = torch.log_softmax(model(token_ids)["logits"], dim=-1)
    if "esm_score" not in fieldnames:
        fieldnames.append("esm_score")
    for row in rows:
        wildtype, sequence_index, mutant = parse_mutation(
            row[mutation_column], sequence, offset_idx
        )
        score = (
            log_probabilities[0, sequence_index + 1, tokenizer.get_idx(mutant)]
            - log_probabilities[0, sequence_index + 1, tokenizer.get_idx(wildtype)]
        )
        row["esm_score"] = str(float(score.item()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but no accelerator is available")
    return torch.device(requested)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run standalone lightweight ESM inference workflows"
    )
    parser.add_argument("--task", required=True, choices=["extract", "fold", "variant"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--sequence")
    parser.add_argument("--dms-input", type=Path)
    parser.add_argument("--mutation-col", default="mutant")
    parser.add_argument("--offset-idx", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main():
    args = create_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = resolve_device(args.device)
    tokenizer = ProteinTokenizer()
    model = ESMInferenceModel().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    if args.task in {"extract", "fold"}:
        if args.fasta is None or not args.fasta.is_file():
            raise ValueError("--fasta is required for extract and fold")
        records = read_fasta(args.fasta)
        if args.task == "extract":
            extract_representations(model, tokenizer, records, args.output, device)
        else:
            fold_structures(model, tokenizer, records, args.output, device)
        return

    if not args.sequence:
        raise ValueError("--sequence is required for variant")
    if args.dms_input is None or not args.dms_input.is_file():
        raise ValueError("--dms-input is required for variant")
    score_variants(
        model,
        tokenizer,
        args.sequence.strip().upper(),
        args.dms_input,
        args.mutation_col,
        args.offset_idx,
        args.output,
        device,
    )


if __name__ == "__main__":
    main()
