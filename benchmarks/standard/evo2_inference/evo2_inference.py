"""Standalone, small Evo2-style nucleotide inference implementation.

The production example uses NeMo/Megatron.  This reference keeps the public
inference contract while making the model small enough for a deterministic
evaluation smoke test and independent of the OneScience package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


VOCAB = "ACGTN|;:_-abcdefghijklmnopqrstuvwxyz0123456789 "
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}


def encode_text(text: str) -> torch.Tensor:
    if not isinstance(text, str) or not text:
        raise ValueError("prompt must be a non-empty string")
    ids = [TOKEN_TO_ID.get(char, TOKEN_TO_ID["N"]) for char in text]
    return torch.tensor(ids, dtype=torch.long)


class Evo2MambaBlock(nn.Module):
    """Compact causal selective state-space block used by Evo2."""

    def __init__(self, hidden_size: int = 32, conv_kernel: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, hidden_size * 2)
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            conv_kernel,
            groups=hidden_size,
            bias=True,
        )
        self.ssm_projection = nn.Linear(hidden_size, hidden_size * 3)
        self.delta_projection = nn.Linear(hidden_size, hidden_size)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, hidden_size + 1).float()))
        self.D = nn.Parameter(torch.ones(hidden_size))
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        mixed, gate = self.input_projection(self.norm(inputs)).chunk(2, dim=-1)
        kernel = self.depthwise_conv.kernel_size[0]
        mixed = nn.functional.pad(mixed.transpose(1, 2), (kernel - 1, 0))
        mixed = nn.functional.silu(self.depthwise_conv(mixed).transpose(1, 2))
        delta_raw, input_scale, output_scale = self.ssm_projection(mixed).chunk(3, dim=-1)
        delta = nn.functional.softplus(self.delta_projection(delta_raw))
        transition = -torch.exp(self.A_log).view(1, -1)
        state = torch.zeros(inputs.shape[0], inputs.shape[2], device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for index in range(inputs.shape[1]):
            decay = torch.exp(delta[:, index] * transition)
            state = decay * state + delta[:, index] * input_scale[:, index] * mixed[:, index]
            outputs.append(output_scale[:, index] * state + self.D * mixed[:, index])
        selective_output = torch.stack(outputs, dim=1) * torch.sigmoid(gate)
        return residual + self.output_projection(selective_output)


class Evo2InferenceModel(nn.Module):
    """A compact causal Mamba language model with the Evo2 interface."""

    def __init__(self, vocab_size: int = len(VOCAB), hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.mamba_layers = nn.ModuleList([Evo2MambaBlock(hidden_size) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 long tensor")
        hidden = self.token_embedding(input_ids)
        for layer in self.mamba_layers:
            hidden = layer(hidden)
        return self.lm_head(self.final_norm(hidden))

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: int | None = None,
    ) -> str:
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be non-negative and temperature positive")
        if top_k < 0 or not 0 <= top_p <= 1:
            raise ValueError("top_k must be non-negative and top_p in [0, 1]")
        if seed is not None:
            torch.manual_seed(seed)
        tokens = encode_text(prompt).unsqueeze(0).to(next(self.parameters()).device)
        for _ in range(max_new_tokens):
            logits = self(tokens)[:, -1, :] / temperature
            if top_k:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered
            if top_p:
                values, indices = torch.sort(logits, descending=True)
                probabilities = torch.softmax(values, dim=-1)
                keep = torch.cumsum(probabilities, dim=-1) <= top_p
                keep[:, 0] = True
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, torch.where(keep, values, filtered.gather(1, indices)))
                logits = filtered
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return "".join(ID_TO_TOKEN[int(token)] for token in tokens[0].tolist())

    @torch.no_grad()
    def score_sequence(self, sequence: str) -> float:
        """Return mean next-token log likelihood for a nucleotide sequence."""
        encoded = encode_text(sequence).unsqueeze(0).to(next(self.parameters()).device)
        if encoded.shape[1] < 2:
            raise ValueError("sequence must contain at least two characters")
        logits = self(encoded[:, :-1])
        log_probs = torch.log_softmax(logits, dim=-1)
        return float(log_probs.gather(-1, encoded[:, 1:].unsqueeze(-1)).mean())


def load_checkpoint(model: nn.Module, checkpoint: str | Path) -> None:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint must be a state_dict or contain model_state_dict")
    model.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evo2-style generation and sequence scoring")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--ckpt-dir", "--checkpoint", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = Evo2InferenceModel().to(device)
    load_checkpoint(model, args.ckpt_dir)
    model.eval()
    with torch.inference_mode():
        generated = model.generate(args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.seed)
        score = model.score_sequence(generated) if len(generated) > 1 else None
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"prompt": args.prompt, "generated_text": generated, "mean_log_probability": score}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
