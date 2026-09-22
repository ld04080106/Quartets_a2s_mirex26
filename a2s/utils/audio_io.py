from __future__ import annotations


def load_audio(path: str, sample_rate: int = 22050):
    import librosa
    return librosa.load(path, sr=sample_rate, mono=True)[0]
