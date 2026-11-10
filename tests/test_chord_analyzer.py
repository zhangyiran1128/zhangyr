from vhcmusic.representation.event_grid import empty_event_grid
from vhcmusic.verification.chord_analyzer import ChordAnalyzer


def test_chord_analyzer_c_major():
    grid = empty_event_grid(1, subdivisions=4, pitch_min=60, pitch_max=72)
    for midi_pitch in [60, 64, 67]:
        idx = midi_pitch - grid.pitch_min
        grid.data[0, 0, 0, idx, 0] = 1
        grid.data[0, 0, 0, idx, 1] = 4
        grid.data[0, 0, 0, idx, 2] = 0.8
    out = ChordAnalyzer().analyze(grid)
    assert out["chords"][0]["chord"] == "C:maj"
