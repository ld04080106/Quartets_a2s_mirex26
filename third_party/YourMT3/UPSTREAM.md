# YourMT3 upstream provenance

This directory is a code-only snapshot of the YourMT3 Hugging Face Space. It is
vendored so training and inference cannot silently change when the upstream
Space is updated.

- Space: <https://huggingface.co/spaces/mimbres/YourMT3>
- Repository type: Hugging Face Space
- Snapshot revision: `5e66c1ea173a8186e0d20432b841d3180cc015b5`
- Original bundle creation: `2026-07-17T06:11:24.194644+00:00`
- Space metadata license declaration: `apache-2.0`
- Related GitHub repository: <https://github.com/mimbres/YourMT3>
- GitHub repository license declaration: `GPL-3.0`

The two upstream locations do not publish the same repository-level license
declaration. Original per-file copyright and license headers are retained. See
the repository-root `THIRD_PARTY_NOTICES.md` before redistribution.

## Local patches

The checked-in source includes the deterministic changes made by
`hpc/stage1/patch_yourmt3_finetune.py`:

1. registration of the Quartets dataset preset;
2. explicit initialization from the configured pretrained checkpoint;
3. a guard that disables unwanted Weights & Biases initialization;
4. a validation-batch-size guard for constrained GPU memory.

The patched locations are marked with `A2S_*` comments. Generated backups,
caches, experiment logs and all checkpoint files are excluded from Git.

The upstream `amt/src/extras/` and `amt/src/tests/` trees are also excluded.
They are not imported by this project's training or inference path, and the
downloaded extras snapshot contained authentication material that must never be
published. This repository contains no upstream OAuth credentials.

The unused `utils/preprocess/preprocess_rnsynth.py` utility is excluded as well:
the pinned upstream file has an indentation error and is unrelated to the
Quartets dataset path used here.
