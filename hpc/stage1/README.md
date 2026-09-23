# Stage 1: YourMT3 deployment and fine-tuning

Only YourMT3 is maintained. All commands are run from the repository root and all paths
are repository-relative, so the checkout can move between local Windows storage and HPC.

## Pinned source

The tested, code-only Space snapshot is committed at `third_party/YourMT3`.
Normal training and inference do not download or update it. See
`third_party/YourMT3/UPSTREAM.md` for its exact revision and local patches.

## Refreshing or transferring the source

On a machine with internet access, either download the Space directly with
`download_yourmt3.sh` or create a transferable archive:

```bash
bash hpc/stage1/download_yourmt3.sh

# Alternative for an offline cluster:
python hpc/stage1/create_yourmt3_bundle.py \
  --download_dir transfer \
  --output transfer/YourMT3_bundle.tar.gz
```

Transfer both the archive and `.sha256` file. On the cluster:

```bash
python hpc/stage1/import_yourmt3_bundle.py \
  --bundle transfer/YourMT3_bundle.tar.gz \
  --source_dir third_party/YourMT3

export YOURMT3_SOURCE_DIR=third_party/YourMT3
```

`deploy_yourmt3.sh` creates or reuses the runtime environment and deliberately
installs the CUDA-matched PyTorch/Torchaudio pair before the upstream dependencies:

```bash
bash hpc/stage1/deploy_yourmt3.sh
```

For a fully offline cluster, build a Linux wheelhouse with
`build_yourmt3_linux_wheelhouse.sh`, or transfer a relocatable environment made
with `pack_environment.sh`. Verify the installed runtime before loading a large
checkpoint:

```bash
python hpc/stage1/check_yourmt3_runtime.py \
  --source_dir "$YOURMT3_SOURCE_DIR" \
  --import_model_helper
```

## Synthetic quartet data

Install FluidSynth and FFmpeg, place licensed `.sf2` files under `soundfonts/`, and edit
the SoundFont list if needed. Generate 120-BPM-centered, ±6% tempo-jittered audio with
velocity variation and 0.5 seconds leading silence:

```bash
bash hpc/stage1/task_prepare_synthetic_quartets.sh \
  configs/stage1_yourmt3_synth_data_pad05.yaml \
  configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml
```

## Fine-tuning

The selected recipe requests four GPUs and does not apply pitch shift:

```bash
bash hpc/stage1/task_yourmt3_finetune.sh \
  configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml
```

If YourMT3 writes only to `lightning_logs`, materialize the chosen checkpoint into its
deterministic experiment path:

```bash
python hpc/stage1/find_yourmt3_checkpoints.py --source_dir "$YOURMT3_SOURCE_DIR"
python hpc/stage1/materialize_yourmt3_checkpoint.py \
  --source_dir "$YOURMT3_SOURCE_DIR" \
  --model_name quartets_yptf_synth_finetune_h800_4gpu_pad05_nops_v1 \
  --checkpoint PATH_TO_SELECTED_CKPT
```

## Inference and evaluation

```bash
export YOURMT3_ENV_PREFIX=yourmt3
bash hpc/stage1/run_yourmt3_synth_finetuned_nops_eval.sh valid
```

Train the learned quartet voice classifier after producing the train predictions:

```bash
bash hpc/stage1/run_yourmt3_synth_finetuned_nops_eval.sh train
bash hpc/stage1/task_voice_assignment.sh configs/stage1_voice_assignment.yaml
```

The adapter requires the configured checkpoint to exist. It intentionally does not pick
the newest file automatically.
