# Third-party notices

The original Quartets A2S code in this repository is licensed under the MIT
License. That license does not relicense third-party software, model weights,
datasets, SoundFonts, or other external assets.

## YourMT3

This repository vendors a code-only YourMT3 Hugging Face Space snapshot under
`third_party/YourMT3`. It retains its upstream copyright headers and is not
covered by this project's MIT License. YourMT3 checkpoints remain external
assets and retain their upstream terms.

- The [YourMT3 GitHub source repository](https://github.com/mimbres/YourMT3)
  contains a GNU General Public License, version 3 (`GPL-3.0`) license file.
- The [YourMT3 Hugging Face Space](https://huggingface.co/spaces/mimbres/YourMT3),
  which is the source selected by `hpc/stage1/create_yourmt3_bundle.py`, declares
  `apache-2.0` in its README metadata.
- The [YourMT3 Hugging Face model repository](https://huggingface.co/mimbres/YourMT3)
  also declares `apache-2.0` in its model-card metadata.

The GitHub code license and Hugging Face metadata are therefore not identical.
`third_party/YourMT3/UPSTREAM.md` records the exact Space revision and local
patches. The project does not claim that its MIT License applies to those files.
Before redistributing a package containing YourMT3 code or weights, confirm with
the upstream maintainer or competition organizer which license grant governs the
exact snapshot and satisfy all applicable notice and source requirements.

## Other external assets

Quartets data, pretrained/fine-tuned checkpoints, GM SoundFonts, FluidSynth,
FFmpeg, PyTorch, and Python dependencies are governed by their respective
licenses or dataset terms. They are intentionally excluded from this source
repository unless explicitly stated otherwise.

This notice summarizes repository metadata for engineering provenance and is
not legal advice.
