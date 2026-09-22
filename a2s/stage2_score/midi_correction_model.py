"""Autoregressive MIDI-token correction model used by Stage 2.

Despite sharing a standard encoder-decoder Transformer architecture, this model
does not generate **kern.  Both source and target vocabularies contain the
project's quantized, voice-aware MIDI tokens.  Deterministic code converts the
corrected tokens to legal **kern afterwards.
"""

from __future__ import annotations

import torch
from torch import nn


class MidiSeq2SeqTransformer(nn.Module):
    """Compact encoder-decoder Transformer for noisy-MIDI to clean-MIDI tokens."""

    def __init__(
        self,
        input_vocab: int,
        output_vocab: int,
        d_model: int = 256,
        nhead: int = 8,
        layers: int = 4,
        max_length: int = 4096,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.input_embedding = nn.Embedding(input_vocab, d_model, padding_idx=0)
        self.output_embedding = nn.Embedding(output_vocab, d_model, padding_idx=0)
        self.position = nn.Embedding(max_length, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=dim_feedforward or d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(d_model, output_vocab)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_padding = source.eq(0)
        target_padding = target.eq(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target.shape[1], device=target.device
        )
        source_positions = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
        target_positions = torch.arange(target.shape[1], device=target.device).unsqueeze(0)
        hidden = self.transformer(
            self.input_embedding(source) + self.position(source_positions),
            self.output_embedding(target) + self.position(target_positions),
            tgt_mask=causal_mask,
            src_key_padding_mask=source_padding,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=source_padding,
        )
        return self.output(hidden)


class MidiCorrectionModel:
    """Checkpoint loader and greedy decoder for the selected Stage 2 model."""

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:  # PyTorch < 2.0 compatibility.
            payload = torch.load(checkpoint, map_location=device)
        self.input_vocab = payload["input_vocab"]
        self.output_vocab = payload["output_vocab"]
        config = payload.get("config", {})
        model_config = config.get("model", {})
        data_config = config.get("data", {})
        max_length = int(max(
            model_config.get("max_length", 4096),
            data_config.get("max_input_tokens", 2048),
            data_config.get("max_output_tokens", 4096),
        ))
        self.max_input_tokens = int(data_config.get("max_input_tokens", max_length))
        self.model = MidiSeq2SeqTransformer(
            len(self.input_vocab),
            len(self.output_vocab),
            int(model_config.get("d_model", 256)),
            int(model_config.get("nhead", 8)),
            int(model_config.get("layers", 4)),
            max_length=max_length,
            dropout=float(model_config.get("dropout", 0.1)),
            dim_feedforward=int(model_config.get("dim_feedforward", 0)) or None,
        ).to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.device = device
        self.inverse_output = {value: key for key, value in self.output_vocab.items()}

    @torch.no_grad()
    def generate(self, input_tokens: list[str], max_tokens: int = 2048) -> list[str]:
        """Greedily decode until EOS or the configured hard length limit."""

        input_limit = max(0, self.max_input_tokens - 2)
        source_ids = [1] + [
            self.input_vocab.get(token, 3) for token in input_tokens[:input_limit]
        ] + [2]
        source = torch.tensor([source_ids], device=self.device)
        target = torch.tensor([[1]], device=self.device)
        for _ in range(max_tokens):
            logits = self.model(source, target)[:, -1]
            next_id = logits.argmax(-1, keepdim=True)
            target = torch.cat((target, next_id), dim=1)
            if int(next_id.item()) == 2:
                break
        return [
            self.inverse_output.get(int(value), "<UNK>")
            for value in target[0, 1:-1]
        ]

