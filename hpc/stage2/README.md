# Stage 2: predicted MIDI events to corrected MIDI events

The maintained Stage 2 is long-v2. It learns from real Stage 1 predictions after the
learned voice assignment model; ground-truth targets are MIDI-like events parsed from
`**kern`. Direct neural `**kern` generation and the old hybrid gate are not maintained.

## Prepare real predicted-event pairs

```bash
bash hpc/stage2/prepare_stage2_train_pred_events.sh train
bash hpc/stage2/prepare_stage2_train_pred_events.sh valid
```

The preparation step runs the selected YourMT3 checkpoint and voice model when outputs
are missing. It then writes:

```text
outputs/stage2_midi_pred_events_data/train.jsonl
outputs/stage2_midi_pred_events_data/valid.jsonl
```

Check token lengths and missing samples before training:

```bash
python scripts/check_stage2_midi_data.py \
  --config configs/stage2_midi_train_pred_events_long_v2.yaml
```

## Train

```bash
bash hpc/stage2/task_stage2_midi_pred_events_train.sh \
  configs/stage2_midi_train_pred_events_long_v2.yaml
```

The selected checkpoint is
`outputs/stage2_midi_pred_events_long_model_v2/best.pt`. Per-epoch checkpoints are kept
locally for model selection but are excluded from Git and from the submission archive.

## Infer and evaluate

```bash
bash hpc/stage2/run_stage2_midi_pred_events_long_v2_from_voice_model.sh valid
```

The neural output is converted to `**kern` by the deterministic rule converter. If neural
inference fails for a sample, the final submission pipeline converts the uncorrected Stage
1 events instead; if that also fails, it emits a legal rest score.
