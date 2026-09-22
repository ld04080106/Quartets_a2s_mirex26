# Resource declaration

Values marked `MEASURE` must be filled from the final packaged run; they are
intentionally not guessed.

## Training data

- Quartet-specific YourMT3 fine-tuning: 24,333 Quartets training excerpts.
- Audio augmentation: score-derived FluidSynth audio, 0.5 s leading silence,
  tempo centered at 120 BPM with ±6% variation, velocity variation, and multiple
  GM SoundFonts.
- Voice assignment: 17,274 matched predicted-note examples from 1,912 excerpts.
- Stage2 MIDI correction: 24,333 real Stage1-prediction/ground-truth MIDI pairs;
  validation uses 6,088 pairs.
- Base checkpoint: pretrained YourMT3/YPTF-MoE multi-task checkpoint. Cite the
  upstream checkpoint and declare its pretraining corpora/license:
  **MEASURE/CONFIRM BEFORE SUBMISSION**.
- No MIREX Quartets test data or undisclosed evaluation data is used for training.

## Model size

- YourMT3: 64.1 M total parameters (46.5 M trainable during fine-tuning).
- Stage2 Transformer: 12,649,176 parameters.
- Voice assignment: histogram gradient boosting model, 2.17 MB serialized.
- Total neural parameters loaded: approximately 76.75 M.

## Training compute

- YourMT3 quartet fine-tuning: 4 × NVIDIA H800; wall time/GPU-hours:
  **MEASURE FROM FINAL TRAINING LOG**.
- Stage2 long-v2: one GPU; approximately 5.68 wall-clock hours; exact GPU model
  and GPU-hours: **CONFIRM**.
- Voice-assignment training device/time: **MEASURE/CONFIRM**.

## Inference compute

- One CUDA GPU, inference batch size 1; bf16 when available, otherwise fp16.
- Internal 8-piece H800 Staves-Informed run: 132.77 seconds end to end,
  507,740,160 bytes peak PyTorch allocation, 100% valid output and exact header
  preservation. This short run includes model startup and is not a full-set
  runtime guarantee.
- Complete evaluation-set time: **MEASURE**.
- Peak allocated VRAM and GPU model: copy from
  `OUTPUT_KERN_DIR/logs/pipeline_report.json` after the final dry run.
- Direct evaluation must remain below 32 GB VRAM and 24 hours on one V100-class
  GPU; otherwise use the precomputed-kern fallback route.

## Software and licensing

- Development runtime: Python 3.11, PyTorch/Torchaudio 2.5.1 CUDA 12.1.
- Original Quartets A2S source code: MIT License.
- YourMT3 GitHub source repository: GPL-3.0. The Hugging Face Space snapshot
  used by the deployment script and the Hugging Face model repository each
  declare Apache-2.0 in repository metadata. These upstream declarations are
  recorded separately because they differ; see `THIRD_PARTY_NOTICES.md`.
- External YourMT3 files and checkpoints retain their upstream terms and are
  not relicensed under MIT. Confirm the applicable grant for the exact bundled
  snapshot before public redistribution.
