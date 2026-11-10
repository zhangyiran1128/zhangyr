from pathlib import Path

import pytest

from vhcmusic.representation.midi_io import event_grid_to_midi, midi_to_event_grid, save_midi


def test_midi_io_roundtrip(tmp_path: Path):
    miditoolkit = pytest.importorskip("miditoolkit")
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    inst = miditoolkit.Instrument(program=0, is_drum=False, name="lead")
    inst.notes.append(miditoolkit.Note(velocity=90, pitch=60, start=0, end=480))
    midi.instruments.append(inst)
    path = tmp_path / "one.mid"
    midi.dump(str(path))
    grid = midi_to_event_grid(path, bars=1, subdivisions=4, pitch_min=60, pitch_max=72)
    assert grid.data[..., 0].sum() >= 1
    out = event_grid_to_midi(grid)
    out_path = save_midi(out, tmp_path / "roundtrip.mid")
    assert out_path.exists()
