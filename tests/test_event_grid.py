import numpy as np

from vhcmusic.representation.event_grid import empty_event_grid, event_grid_to_pianoroll, grid_difference, pianoroll_to_event_grid


def test_event_grid_roundtrip_pianoroll():
    grid = empty_event_grid(2, subdivisions=4, pitch_min=60, pitch_max=71)
    grid.data[0, 0, 1, 0, 0] = 1
    grid.data[0, 0, 1, 0, 1] = 2
    grid.data[0, 0, 1, 0, 2] = 0.8
    roll = event_grid_to_pianoroll(grid)
    assert roll.shape == (6, 8, 12)
    assert roll[0, 1, 0] == 1
    assert roll[0, 2, 0] == 1
    back = pianoroll_to_event_grid(roll, bars=2, subdivisions=4, pitch_min=60, pitch_max=71)
    assert back.data[..., 0].sum() >= 1


def test_grid_difference_shape():
    a = empty_event_grid(2)
    b = a.copy()
    b.data[0, 0, 0, 0, 0] = 1
    diff = grid_difference(a, b)
    assert diff.shape == (6, 2)
    assert diff[0, 0] > 0
