# MIREX 2026 Audio-to-Score submission

This package implements both MIREX 2026 input modes:

- Blind A2S: audio only.
- Staves-Informed A2S: audio plus a file/directory containing the supplied
  `**kern` header metadata.

Pipeline:

```text
audio
  -> quartet-fine-tuned YourMT3 (0.5 s leading pad)
  -> learned four-voice assignment
  -> long-context MIDI-to-MIDI correction Transformer
  -> deterministic quantization and **kern writer
  -> supplied-header projection (staves-informed only)
  -> validation, repair, and legal fallback
```

## Entry point

From the package root:

```bash
bash transcription.sh INPUT_AUDIO_OR_DIR OUTPUT_KERN_DIR [METADATA_FILE_OR_DIR]
```

The input may be one FLAC/WAV file or a directory searched recursively. One
`<audio-basename>.krn` is written per unique input basename. Metadata may be one
header file or a directory whose files share the audio basenames. `.krn`,
`.kern`, `.txt`, and `.json` metadata are supported. JSON may contain
`header`/`kern_header`, or separate meter/key/tempo fields.

The evaluator does not need to edit model paths. They are resolved relative to
the package through `configs/pipeline_submission.yaml`. Per-sample exceptions do
not abort the batch; a valid rest score is emitted and the exception is recorded
in `OUTPUT_KERN_DIR/logs/failed_samples.csv`. A missing required model asset is
a preflight error instead of silently producing an empty batch.

If a required checkpoint is absent, `transcription.sh` invokes the checksum-
pinned asset downloader before preflight. Public OSS URLs and verified SHA-256
values are included in the final config; the six environment variables in the
root README optionally select a mirror. Existing files are never downloaded
again, but configured checksums are still verified. For authenticated HTTPS,
set `A2S_ASSET_BEARER_TOKEN` without storing the token in the repository.

## Environment

Python 3.11 and one CUDA GPU are expected. Install PyTorch/Torchaudio for the
host CUDA version first, then:

```bash
python -m pip install -r requirements_submission.txt
python scripts/ensure_model_assets.py --config configs/pipeline_submission.yaml
python scripts/check_submission_assets.py --config configs/pipeline_submission.yaml
```

Precision is automatic: bf16 on supported GPUs, fp16 otherwise. Total and
per-stage runtime, GPU name, and peak PyTorch-allocated memory are saved in
`logs/pipeline_report.json`. The final config removes temporary MIDI/events after
each run; set `pipeline.save_intermediates: true` only in a copied debug config.

The helper `hpc/submission/setup_submission_env.sh` installs the non-PyTorch
runtime requirements after the CUDA-compatible PyTorch/Torchaudio pair has been
installed. `hpc/submission/run_submission_dryrun.sh` runs the same root entry
point with an optional sample limit.

## Smoke and full runs

```bash
A2S_LIMIT=8 bash transcription.sh INPUT_AUDIO_DIR OUTPUT_SMOKE_DIR METADATA_DIR
python scripts/check_submission_outputs.py \
  --input_audio INPUT_AUDIO_DIR \
  --output_dir OUTPUT_SMOKE_DIR \
  --out_json OUTPUT_SMOKE_DIR/logs/submission_contract.json

bash transcription.sh INPUT_AUDIO_DIR OUTPUT_FULL_DIR METADATA_DIR
```

Omit `METADATA_DIR` for Blind A2S. Set `A2S_PYTHON=/path/to/python` if needed.
For diagnostics, copy the config and set `pipeline.save_intermediates: true`.

## Package

Required packaged assets are the selected YourMT3 source/checkpoint,
`outputs/voice_assignment/model.pkl`, and
`outputs/stage2_midi_pred_events_long_model_v2/best.pt`.

Build the final archive on the machine containing all assets:

```bash
python scripts/package_submission.py \
  --config configs/pipeline_submission.yaml \
  --out outputs/mirex2026_a2s_submission \
  --include_runtime_assets \
  --format tar.gz
```

The packager uses an explicit runtime-file whitelist, so training configs,
internal evaluation tools, tests, and historical experiments do not enter the
archive. It also writes a SHA-256 checksum. Before sending the archive, fill in
the remaining measured values in `RESOURCE_DECLARATION.md` and
smoke-test the unpacked package once.

## Runtime-limit fallback

MIREX limits direct evaluation to 32 GB VRAM and 24 hours on one V100-class GPU.
The observed development runs were already roughly 4 hours for Stage1 and 5.5
hours for Stage2 on 6,088 files on faster hardware, so a full V100 benchmark is
mandatory. If the projected official-set time exceeds 24 hours, use the
officially accepted pre-computed-kern route:

```bash
bash transcription.sh EVAL_AUDIO_DIR PRECOMPUTED_KERN_DIR METADATA_DIR
python scripts/check_submission_outputs.py \
  --input_audio EVAL_AUDIO_DIR \
  --output_dir PRECOMPUTED_KERN_DIR
python scripts/package_precomputed_kern.py \
  --input_audio EVAL_AUDIO_DIR \
  --kern_dir PRECOMPUTED_KERN_DIR \
  --out outputs/mirex2026_precomputed_kern
```
