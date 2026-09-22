"""Thin, deterministic wrapper around the upstream YourMT3 inference code.

The adapter stages an explicitly configured checkpoint into the directory layout
expected by YourMT3.  It never searches for a "latest" checkpoint: silently
selecting another experiment would make a competition run irreproducible.
"""
from __future__ import annotations

import os
import shutil
import sys
from collections import Counter
from pathlib import Path


MODEL_ARGUMENTS = {
    "yptf_moe_multi_nops": [
        "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt",
        "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
        "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe",
        "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu",
        "-epe", "rope", "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1",
    ],
    "quartets_yptf_synth_finetune_h800_4gpu_pad05_nops_v1": [
        "quartets_yptf_synth_finetune_h800_4gpu_pad05_nops_v1@last.ckpt",
        "-p", "2026_quartets", "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
        "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe",
        "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu",
        "-epe", "rope", "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1",
    ],
}


def _checkpoint_target_from_args(source_dir: Path, args: list[str]) -> Path:
    exp_id = args[0]
    project = "ymt3"
    if "-p" in args:
        index = args.index("-p")
        if index + 1 < len(args):
            project = args[index + 1]
    if "@" in exp_id:
        exp_id, checkpoint_name = exp_id.split("@", 1)
    else:
        checkpoint_name = "last.ckpt"
    return source_dir / "amt" / "logs" / project / exp_id / "checkpoints" / checkpoint_name


def _stage_external_checkpoint(source_dir: Path, args: list[str], checkpoint_path: str | Path | None) -> None:
    if not checkpoint_path:
        return
    source = Path(os.path.expandvars(os.path.expanduser(str(checkpoint_path))))
    if not source.is_absolute():
        source = source_dir / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"fine-tuned YourMT3 checkpoint not found: {source}")
    target = _checkpoint_target_from_args(source_dir, args)
    if target.resolve() == source:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return
    shutil.copy2(source, target)


class YourMT3:
    def __init__(
        self,
        source_dir: str | Path,
        model_name: str = "yptf_moe_multi_nops",
        precision: str = "16",
        device: str = "cuda",
        inference_batch_size: int = 1,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        if not (self.source_dir / "model_helper.py").exists():
            raise FileNotFoundError(f"YourMT3 Space snapshot is incomplete: {self.source_dir}")
        sys.path.insert(0, str(self.source_dir / "amt" / "src"))
        sys.path.insert(0, str(self.source_dir))
        from model_helper import load_model_checkpoint
        from utils.audio import slice_padded_array
        from utils.event2note import merge_zipped_note_events_and_ties_to_notes
        from utils.note2event import mix_notes
        from utils.utils import write_model_output_as_midi

        if model_name not in MODEL_ARGUMENTS:
            raise ValueError(f"unsupported YourMT3 model preset: {model_name}")
        args = [*MODEL_ARGUMENTS[model_name], "-pr", precision]
        _stage_external_checkpoint(self.source_dir, args, checkpoint_path)
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be >= 1")
        self.inference_batch_size = inference_batch_size
        self.device = device
        self.slice_padded_array = slice_padded_array
        self.merge_note_events = merge_zipped_note_events_and_ties_to_notes
        self.mix_notes = mix_notes
        self.write_model_output_as_midi = write_model_output_as_midi
        self._old_cwd = Path.cwd()
        os.chdir(self.source_dir)
        try:
            self.model = load_model_checkpoint(args=args, device="cpu")
            self.model.to(device)
        finally:
            os.chdir(self._old_cwd)

    def _transcribe(self, audio_path: Path, track_name: str) -> Path:
        """Memory-bounded variant of the Space's model_helper.transcribe()."""
        import gc
        import torch
        import torchaudio

        audio_segments = pred_token_arr = None
        try:
            audio, sample_rate = torchaudio.load(uri=str(audio_path))
            audio = torch.mean(audio, dim=0).unsqueeze(0)
            audio = torchaudio.functional.resample(
                audio, sample_rate, self.model.audio_cfg["sample_rate"]
            )
            audio_segments_np = self.slice_padded_array(
                audio, self.model.audio_cfg["input_frames"], self.model.audio_cfg["input_frames"]
            )
            audio_segments = torch.from_numpy(audio_segments_np.astype("float32"))
            audio_segments = audio_segments.to(self.device).unsqueeze(1)
            with torch.inference_mode():
                pred_token_arr, _ = self.model.inference_file(
                    bsz=self.inference_batch_size, audio_segments=audio_segments
                )

            num_channels = self.model.task_manager.num_decoding_channels
            num_items = audio_segments.shape[0]
            start_secs = [
                self.model.audio_cfg["input_frames"] * i / self.model.audio_cfg["sample_rate"]
                for i in range(num_items)
            ]
            notes_by_channel = []
            error_counts = Counter()
            for channel in range(num_channels):
                token_batches = [array[:, channel, :] for array in pred_token_arr]
                zipped, _, _ = self.model.task_manager.detokenize_list_batches(
                    token_batches, start_secs, return_events=True
                )
                channel_notes, channel_errors = self.merge_note_events(zipped)
                notes_by_channel.append(channel_notes)
                error_counts += channel_errors
            notes = self.mix_notes(notes_by_channel)
            self.write_model_output_as_midi(
                notes, "./", track_name, self.model.midi_output_inverse_vocab
            )
            produced = Path("model_output") / f"{track_name}.mid"
            if not produced.exists():
                raise RuntimeError(f"YourMT3 did not create MIDI: {produced}")
            return produced.resolve()
        finally:
            del audio_segments, pred_token_arr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe_to_midi(self, audio_path: str | Path, output_path: str | Path) -> Path:
        source = Path(audio_path).resolve()
        track_name = Path(output_path).stem
        os.chdir(self.source_dir)
        try:
            produced = self._transcribe(source, track_name)
        finally:
            os.chdir(self._old_cwd)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, target)
        return target
