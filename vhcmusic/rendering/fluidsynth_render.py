"""Deterministic per-role FluidSynth rendering and mixing."""

import copy
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from vhcmusic.representation.midi_io import read_midi, save_midi
from vhcmusic.representation.track_mapping import GM_PROGRAMS, TrackMapper
from .stem_mixing import mix_stems

ROLE_GAIN_DB = {"lead": -2, "bass": -4, "rhythm": -5, "harmony_pad": -6, "accompaniment": -5, "drums": -3}
ROLE_PAN = {"lead": 0, "bass": 0, "rhythm": -0.1, "harmony_pad": -0.18, "accompaniment": 0.18, "drums": 0}


def check_fluidsynth(fluidsynth_bin="fluidsynth", soundfont=None):
    if shutil.which(fluidsynth_bin) is None:
        raise FileNotFoundError("Set FLUIDSYNTH_BIN or add FluidSynth to PATH")
    if not soundfont or not Path(soundfont).is_file():
        raise FileNotFoundError("Set VHCMUSIC_SOUNDFONT to a GM-compatible SF2 file")


def render_profile(settings):
    soundfont = settings.get("soundfont") or os.environ.get("VHCMUSIC_SOUNDFONT")
    executable = os.environ.get("FLUIDSYNTH_BIN", settings.get("fluidsynth_bin", "fluidsynth"))
    check_fluidsynth(executable, soundfont)
    font = hashlib.sha256()
    with Path(soundfont).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            font.update(chunk)
    version = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True)
    return {"soundfont_sha256": font.hexdigest(), "fluidsynth_version": version.stdout.strip(),
            "settings": {key: value for key, value in settings.items() if key not in {"soundfont", "fluidsynth_bin"}}}


def render_midi_to_wav(midi_path, output_wav, soundfont=None, fluidsynth_bin="fluidsynth",
                       sample_rate=44100, gain=0.7, normalize_lufs=-14.0, roles_to_gm_program=None,
                       gain_db=None, pan=None, duration_seconds=None):
    import soundfile as sf

    soundfont = soundfont or os.environ.get("VHCMUSIC_SOUNDFONT")
    fluidsynth_bin = os.environ.get("FLUIDSYNTH_BIN", fluidsynth_bin)
    check_fluidsynth(fluidsynth_bin, soundfont)
    midi = read_midi(midi_path)
    out = Path(output_wav)
    stems_dir = out.parent / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    mapper = TrackMapper()
    programs = roles_to_gm_program or GM_PROGRAMS
    gains, pans = gain_db or ROLE_GAIN_DB, pan or ROLE_PAN
    samples = round(duration_seconds * sample_rate) if duration_seconds is not None else None
    if samples is not None and samples < 1:
        raise ValueError('Audio duration must be positive')
    stems = []
    stem_params = []
    with TemporaryDirectory() as directory:
        for role in mapper.roles:
            part = copy.copy(midi)
            part.instruments = [copy.deepcopy(i) for i in midi.instruments
                                if i.notes and mapper.infer_role(i.name, i.program, i.is_drum) == role]
            stem = stems_dir / f"{role}.wav"
            if part.instruments:
                for instrument in part.instruments:
                    instrument.program = programs[role]
                    instrument.is_drum = role == "drums"
                stem_midi = save_midi(part, Path(directory) / f"{role}.mid")
                result = subprocess.run([fluidsynth_bin, "-ni", "-R", "0", "-C", "0", "-r", str(sample_rate),
                                         "-g", str(gain), "-T", "wav", "-F", str(stem),
                                         str(soundfont), str(stem_midi)], capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FluidSynth failed for {role}: {result.stderr}")
                audio, _ = sf.read(stem, frames=samples if samples is not None else -1, dtype="float32", always_2d=True)
            else:
                audio = np.zeros((samples or 1, 2), dtype=np.float32)
            if samples is not None and len(audio) < samples:
                audio = np.pad(audio, ((0, samples - len(audio)), (0, 0)))
            sf.write(stem, audio, sample_rate, subtype="PCM_16")
            stems.append(stem)
            stem_params.append((gains[role], pans[role]))
    return mix_stems(stems, out, target_lufs=normalize_lufs, stem_params=stem_params)
