# Quartets A2S — MIREX 2026 Two-Stage System

This repository contains a MIREX 2026 Audio-to-Score Transcription submission.
The maintained system uses YourMT3; the discontinued official MT3/T5X adapter
and the neural model that generated `**kern` directly are not included.

```text
audio
  → quartet-fine-tuned YourMT3
  → unified note events
  → learned violin_1 / violin_2 / viola / cello assignment
  → MIDI-token correction Transformer
  → deterministic quantization and **kern writer
  → supplied-header projection (Staves-Informed)
  → validation / repair / legal fallback
```

The neural Stage 2 consumes and produces voice-aware MIDI-like tokens. A
deterministic writer always generates the final `**kern`, so every input audio
file receives a structurally legal output. The rule-based converter also serves
as the fallback if neural correction fails.

## Repository layout

```text
a2s/                  reusable data, AMT post-processing, Stage 2 and evaluation code
configs/              selected final configs only
hpc/stage1/           YourMT3 offline deployment, fine-tuning and inference
hpc/stage2/           predicted-event data preparation and MIDI correction training
hpc/submission/       submission build and dry-run helpers
scripts/              data, training, evaluation and packaging entry points
third_party/YourMT3/  pinned, code-only YourMT3 Space snapshot
tests/                 unit tests for parsing, events, voices and **kern validity
```

`data/`, `outputs/`, `transfer/`, `hpc_assets/`, `internal_eval/`, checkpoints and
SoundFonts are intentionally excluded by `.gitignore`. They may remain on a
workstation or cluster, but are never copied into the public source repository.
The final competition archive copies only an explicit runtime whitelist plus the
selected model assets.

## Environment

Data preparation and unit tests:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Competition inference needs Python 3.11, one CUDA GPU, and the dependencies in
`requirements_submission.txt`. Install the PyTorch/Torchaudio pair matching the host
CUDA runtime first. The exact YourMT3 code snapshot used by this system is pinned under
`third_party/YourMT3`; model checkpoints remain external assets.

## 1. Dataset and oracle events

Edit `configs/data_quartets.yaml`, then run:

```bash
python scripts/prepare_quartets.py --config configs/data_quartets.yaml
python scripts/extract_oracle_events.py --config configs/data_quartets.yaml
python scripts/check_oracle_events.py --config configs/data_quartets.yaml
python scripts/deduplicate_manifests.py --config configs/data_quartets.yaml
```

The parser supports four `**kern` spines among auxiliary spines, reciprocal and dotted
durations, rests/null tokens, common accidentals and octave spelling, chords, barlines,
meters, initial key signatures and tie markers. It does not implement general Humdrum
spine split/join semantics, tuplets, grace timing or full ornament semantics.

## 2. Stage 1 — YourMT3

Offline download/import and HPC details are in
[`hpc/stage1/README.md`](hpc/stage1/README.md). The selected training recipe is:

```bash
bash hpc/stage1/task_prepare_synthetic_quartets.sh \
  configs/stage1_yourmt3_synth_data_pad05.yaml \
  configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml

bash hpc/stage1/task_yourmt3_finetune.sh \
  configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml
```

The synthetic audio recipe uses score-derived FluidSynth rendering, tempo 120 BPM with
±6% jitter, velocity variation and 0.5 seconds of leading silence. The selected
checkpoint is evaluated with:

```bash
export YOURMT3_ENV_PREFIX=yourmt3
export YOURMT3_SOURCE_DIR=third_party/YourMT3
bash hpc/stage1/run_yourmt3_synth_finetuned_nops_eval.sh valid
```

Train and apply the quartet voice classifier:

```bash
bash hpc/stage1/task_voice_assignment.sh configs/stage1_voice_assignment.yaml
```

## 3. Stage 2 — MIDI correction

Stage 2 is trained on real Stage 1 predictions after learned voice assignment, paired
with MIDI-like targets extracted from ground-truth `**kern`:

```bash
bash hpc/stage2/prepare_stage2_train_pred_events.sh train
bash hpc/stage2/prepare_stage2_train_pred_events.sh valid
bash hpc/stage2/task_stage2_midi_pred_events_train.sh \
  configs/stage2_midi_train_pred_events_long_v2.yaml
```

Inference/evaluation:

```bash
bash hpc/stage2/run_stage2_midi_pred_events_long_v2_from_voice_model.sh valid
```

The selected long-v2 model obtained body WER `0.4946` and body LER `0.8305` on the
6,088-piece validation split. These are internal diagnostics, not official MIREX scores.
The eight-piece end-to-end Staves-Informed dry run obtained 100% valid outputs and exact
header preservation, with body WER `0.4557` and body LER `0.7888`.

## 4. Submission

Required local assets are declared in `configs/pipeline_submission.yaml`:

- the external YourMT3 source snapshot and selected fine-tuned checkpoint;
- `outputs/voice_assignment/model.pkl`;
- `outputs/stage2_midi_pred_events_long_model_v2/best.pt`.

The final config contains immutable public OSS URLs and verified SHA-256 values.
The following optional environment variables override those defaults when
mirroring the assets elsewhere:

```bash
export A2S_STAGE1_CHECKPOINT_URL=https://MIRROR/PATH/yourmt3.ckpt
export A2S_STAGE1_CHECKPOINT_SHA256=SHA256_OF_MIRRORED_FILE
export A2S_VOICE_MODEL_URL=https://MIRROR/PATH/voice_assignment.pkl
export A2S_VOICE_MODEL_SHA256=SHA256_OF_MIRRORED_FILE
export A2S_STAGE2_CHECKPOINT_URL=https://MIRROR/PATH/stage2_best.pt
export A2S_STAGE2_CHECKPOINT_SHA256=SHA256_OF_MIRRORED_FILE
```

`transcription.sh` checks local files first, downloads only missing assets to a
temporary file, verifies SHA-256, and atomically installs them. Use
`A2S_ASSET_BEARER_TOKEN` only at runtime if the server requires authentication.

Validate and build:

```bash
python scripts/ensure_model_assets.py --config configs/pipeline_submission.yaml
python scripts/check_submission_assets.py --config configs/pipeline_submission.yaml
bash hpc/submission/build_submission.sh
```

Run the same public entry point used by the evaluator:

```bash
bash transcription.sh INPUT_AUDIO_OR_DIR OUTPUT_KERN_DIR [METADATA_FILE_OR_DIR]
```

Then verify the one-audio/one-valid-kern contract:

```bash
python scripts/check_submission_outputs.py \
  --input_audio INPUT_AUDIO_OR_DIR \
  --output_dir OUTPUT_KERN_DIR
```

See [`SUBMISSION.md`](SUBMISSION.md) for evaluator-facing instructions and
[`RESOURCE_DECLARATION.md`](RESOURCE_DECLARATION.md) for the final compute,
training-data and licensing declaration.

## Generated artifacts

Training produces the following local, Git-ignored assets:

```text
data/manifests_dedup/
data/oracle_events/
data/synthetic_quartets_pad05/
data/yourmt3_synth_pad05_quartets/
third_party/YourMT3/amt/logs/                 # local checkpoints; ignored by Git
outputs/voice_assignment/model.pkl
outputs/stage2_midi_pred_events_data/{train,valid}.jsonl
outputs/stage2_midi_pred_events_long_model_v2/best.pt
```

The final pipeline uses `NoteEventSequence` as the boundary between YourMT3,
voice assignment, MIDI correction and deterministic notation. Model loading is
fail-fast, while individual samples are isolated: failed neural correction falls
back to the Stage 1 events, and a remaining conversion failure emits a valid rest
score and records the exception under the output `logs/` directory.

## Release checklist

- Do not commit datasets, SoundFonts, pretrained checkpoints or experiment outputs.
- This repository's original code is MIT licensed; retain `LICENSE` in releases.
- Retain `THIRD_PARTY_NOTICES.md` and all upstream notices in submission archives.
- Resolve the documented YourMT3 GitHub/Hugging Face license-metadata difference
  before publicly redistributing a bundle containing YourMT3 code or weights.
- Fill remaining measured values in `RESOURCE_DECLARATION.md`.
- Build in a clean directory and run at least the 8-file Staves-Informed diagnostic.
- Benchmark enough files to demonstrate compliance with the MIREX runtime limit.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for external component
licenses. The repository MIT license does not apply to YourMT3, datasets,
checkpoints, SoundFonts, or other third-party assets.
