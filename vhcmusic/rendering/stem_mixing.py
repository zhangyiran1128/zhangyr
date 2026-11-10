"""Fixed role gain, stereo pan, bus limiting and loudness normalization."""

from pathlib import Path

import numpy as np
import soundfile as sf

from .loudness import loudness_normalize, peak_normalize


def mix_stems(stem_paths, output_path, target_lufs=-14.0, stem_params=None):
    if not stem_paths:
        raise ValueError("No stems to mix")
    params = stem_params or [(0, 0)] * len(stem_paths)
    stems = [sf.read(path, dtype="float32", always_2d=True) for path in stem_paths]
    sr = stems[0][1]
    if any(rate != sr for _, rate in stems):
        raise ValueError("All stems must share a sample rate")
    mix = np.zeros((max(len(audio) for audio, _ in stems), 2), dtype=np.float32)
    for (stem, _), (db, pan) in zip(stems, params, strict=True):
        angle = (float(pan) + 1) * np.pi / 4
        stereo = stem if stem.shape[1] == 2 else np.repeat(stem, 2, axis=1)
        mix[:len(stem)] += stereo * (10 ** (db / 20)) * np.sqrt(2) * np.array([np.cos(angle), np.sin(angle)])
    if np.max(np.abs(mix)) > 0:
        if np.max(np.abs(mix)) > 0.98:
            mix = peak_normalize(mix)
        if len(mix) >= int(0.4 * sr):
            mix = loudness_normalize(mix, sr, target_lufs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, mix, sr, subtype="PCM_16")
    return out
