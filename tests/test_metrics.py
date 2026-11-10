import numpy as np

from vhcmusic.metrics.control_metrics import active_role_f1, rhythm_mae, section_boundary_f1
from vhcmusic.metrics.editing_metrics import target_success_rate


def test_control_metrics():
    assert section_boundary_f1([5, 9], [4, 9], tolerance=1)["f1"] == 1.0
    assert rhythm_mae([0.1, 0.3], [0.2, 0.1]) > 0
    assert active_role_f1(np.array([[1, 0]]), np.array([[1, 1]]))["f1"] < 1


def test_target_success_counts_only_applicable_explicit_constraints():
    assert target_success_rate([True, False, None, True]) == 2 / 3
    assert target_success_rate([None]) is None
