# Third-party notices

The original Quartets A2S code in this repository is licensed under the MIT
License. That license does not relicense third-party software, model weights,
datasets, SoundFonts, or other external assets.

## YourMT3

This project can load an external YourMT3 snapshot and checkpoint. Those files
are not part of the Git repository and retain their upstream licenses and
copyright notices.

- The [YourMT3 GitHub source repository](https://github.com/mimbres/YourMT3)
  contains a GNU General Public License, version 3 (`GPL-3.0`) license file.
- The [YourMT3 Hugging Face Space](https://huggingface.co/spaces/mimbres/YourMT3),
  which is the source selected by `hpc/stage1/create_yourmt3_bundle.py`, declares
  `apache-2.0` in its README metadata.
- The [YourMT3 Hugging Face model repository](https://huggingface.co/mimbres/YourMT3)
  also declares `apache-2.0` in its model-card metadata.

The GitHub code license and Hugging Face metadata are therefore not identical.
The bundling scripts record which Hugging Face repository type and revision was
downloaded, preserve upstream files, and do not claim that the MIT license
applies to those files. Before redistributing a package that contains YourMT3
code or weights, confirm with the upstream maintainer or competition organizer
which license grant governs the exact snapshot being distributed and satisfy
all applicable notice and source-distribution requirements.

## Other external assets

Quartets data, pretrained/fine-tuned checkpoints, GM SoundFonts, FluidSynth,
FFmpeg, PyTorch, and Python dependencies are governed by their respective
licenses or dataset terms. They are intentionally excluded from this source
repository unless explicitly stated otherwise.

This notice summarizes repository metadata for engineering provenance and is
not legal advice.
