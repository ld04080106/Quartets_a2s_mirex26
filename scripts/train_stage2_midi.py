from __future__ import annotations

import argparse
import math
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import save_json
from a2s.utils.logging import configure_logging
from a2s.utils.seed import seed_everything


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_jsonl(path: Path) -> list[tuple[list[str], list[str], dict[str, Any]]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append((
                list(item["source_tokens"]),
                list(item["target_tokens"]),
                {"sample_id": item.get("sample_id"), "source": item.get("source"), "line_no": line_no},
            ))
    return rows


def _vocab(sequences: list[list[str]]) -> dict[str, int]:
    values = sorted(set(token for sequence in sequences for token in sequence))
    return {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, **{token: i + 4 for i, token in enumerate(values)}}


def _encode(tokens: list[str], vocab: dict[str, int], limit: int) -> list[int]:
    trimmed = [token for token in tokens if token not in {"<BOS_MIDI>", "<EOS_MIDI>"}]
    return [1] + [vocab.get(token, 3) for token in trimmed[:limit - 2]] + [2]


def _batch(examples, input_vocab, output_vocab, max_input, max_output, device):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    source = pad_sequence([
        torch.tensor(_encode(x, input_vocab, max_input), dtype=torch.long)
        for x, _, _ in examples
    ], batch_first=True).to(device)
    target = pad_sequence([
        torch.tensor(_encode(y, output_vocab, max_output), dtype=torch.long)
        for _, y, _ in examples
    ], batch_first=True).to(device)
    return source, target


def _make_scheduler(optimizer, config: dict, total_steps: int):
    import torch

    name = str(deep_get(config, "training.scheduler", "none")).lower()
    if name in {"", "none", "constant"}:
        return None
    warmup_steps = int(deep_get(config, "training.warmup_steps", 0))
    warmup_ratio = float(deep_get(config, "training.warmup_ratio", 0.0))
    if warmup_steps <= 0 and warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)
    min_lr_ratio = float(deep_get(config, "training.min_lr_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        if name in {"linear_warmup", "linear"}:
            remaining = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
            return max(min_lr_ratio, 1.0 - progress)
        if name in {"cosine", "linear_warmup_cosine", "cosine_warmup"}:
            remaining = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"unsupported training.scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _run_epoch(model, examples, input_vocab, output_vocab, config, device, optimizer=None, scheduler=None) -> float:
    import torch

    model.train(optimizer is not None)
    batch_size = int(deep_get(config, "training.batch_size", 8))
    max_input = int(deep_get(config, "data.max_input_tokens", 2048))
    max_output = int(deep_get(config, "data.max_output_tokens", 2048))
    grad_clip = float(deep_get(config, "training.grad_clip_norm", 1.0))
    grad_accum_steps = max(1, int(deep_get(config, "training.grad_accum_steps", 1)))
    label_smoothing = float(deep_get(config, "training.label_smoothing", 0.0))
    use_amp = bool(deep_get(config, "training.amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    losses = []
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(examples), batch_size):
        source, target = _batch(examples[start:start + batch_size], input_vocab, output_vocab, max_input, max_output, device)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(source, target[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target[:, 1:].reshape(-1),
                    ignore_index=0,
                    label_smoothing=label_smoothing,
                )
                loss_for_backward = loss / grad_accum_steps
        if optimizer is not None:
            scaler.scale(loss_for_backward).backward()
            step_index = start // batch_size
            should_step = ((step_index + 1) % grad_accum_steps == 0) or (start + batch_size >= len(examples))
            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage2 MIDI-to-MIDI correction model.")
    parser.add_argument("--config", default="configs/stage2_midi_train_pred_events_long_v2.yaml")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    seed = int(deep_get(config, "training.seed", 1337))
    seed_everything(seed)

    import torch
    from a2s.stage2_score.midi_correction_model import MidiSeq2SeqTransformer

    output_dir = _project_path(project_root, deep_get(config, "training.output_dir", "outputs/stage2_midi_model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(deep_get(config, "training.log_file", str(output_dir / "train.log")))
    train = _load_jsonl(_project_path(project_root, deep_get(config, "data.train_jsonl", "outputs/stage2_midi_data/train.jsonl")))
    valid_path = deep_get(config, "data.valid_jsonl", "outputs/stage2_midi_data/valid.jsonl")
    valid = _load_jsonl(_project_path(project_root, valid_path)) if valid_path else []
    input_vocab = _vocab([x for x, _, _ in train + valid])
    output_vocab = _vocab([y for _, y, _ in train + valid])
    max_length = int(max(
        deep_get(config, "model.max_length", 2048),
        deep_get(config, "data.max_input_tokens", 2048),
        deep_get(config, "data.max_output_tokens", 2048),
    ))
    model = MidiSeq2SeqTransformer(
        len(input_vocab), len(output_vocab),
        int(deep_get(config, "model.d_model", 256)),
        int(deep_get(config, "model.nhead", 8)),
        int(deep_get(config, "model.layers", 4)),
        max_length=max_length,
        dropout=float(deep_get(config, "model.dropout", 0.1)),
        dim_feedforward=int(deep_get(config, "model.dim_feedforward", 0)) or None,
    )
    device_name = deep_get(config, "training.device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_get(config, "training.learning_rate", 3e-4)),
        weight_decay=float(deep_get(config, "training.weight_decay", 0.01)),
    )
    batch_size = int(deep_get(config, "training.batch_size", 8))
    grad_accum_steps = max(1, int(deep_get(config, "training.grad_accum_steps", 1)))
    epochs = int(deep_get(config, "training.epochs", 20))
    total_steps = math.ceil(len(train) / max(1, batch_size * grad_accum_steps)) * epochs
    scheduler = _make_scheduler(optimizer, config, total_steps)
    best = float("inf")
    best_epoch = 0
    history = []
    patience = int(deep_get(config, "training.early_stop_patience", 0))
    min_delta = float(deep_get(config, "training.early_stop_min_delta", 0.0))
    save_every_epoch = bool(deep_get(config, "training.save_every_epoch", False))
    epoch_dir = output_dir / "epoch_checkpoints"
    if save_every_epoch:
        epoch_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Stage2 MIDI training: train=%d valid=%d input_vocab=%d output_vocab=%d batch=%d accum=%d total_steps=%d",
        len(train), len(valid), len(input_vocab), len(output_vocab), batch_size, grad_accum_steps, total_steps,
    )
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(train)
        train_loss = _run_epoch(model, train, input_vocab, output_vocab, config, device, optimizer, scheduler)
        valid_loss = _run_epoch(model, valid, input_vocab, output_vocab, config, device) if valid else train_loss
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss, "learning_rate": current_lr})
        payload = {"model": model.state_dict(), "config": config, "input_vocab": input_vocab, "output_vocab": output_vocab}
        torch.save(payload, output_dir / "last.pt")
        if save_every_epoch:
            torch.save(payload, epoch_dir / f"epoch_{epoch + 1:03d}_valid_{valid_loss:.6f}.pt")
        if valid_loss <= best - min_delta:
            best = valid_loss
            best_epoch = epoch + 1
            torch.save(payload, output_dir / "best.pt")
            torch.save(payload, output_dir / "model.pt")
        logger.info("epoch=%d train_loss=%.6f valid_loss=%.6f lr=%.8f best=%.6f@%d", epoch + 1, train_loss, valid_loss, current_lr, best, best_epoch)
        if patience > 0 and best_epoch > 0 and epoch + 1 - best_epoch >= patience:
            logger.info("early stopping at epoch=%d best_epoch=%d best_valid_loss=%.6f", epoch + 1, best_epoch, best)
            break
    save_json(output_dir / "train_report.json", {
        "num_train_examples": len(train),
        "num_valid_examples": len(valid),
        "train_source_counts": dict(Counter(meta.get("source", "unknown") for _, _, meta in train)),
        "valid_source_counts": dict(Counter(meta.get("source", "unknown") for _, _, meta in valid)),
        "input_vocab_size": len(input_vocab),
        "output_vocab_size": len(output_vocab),
        "best_valid_loss": best,
        "best_epoch": best_epoch,
        "history": history,
    })


if __name__ == "__main__":
    main()
