from vhcmusic.editing.edit_mask import mask_from_constraints, mask_from_request
from vhcmusic.planning.schema import make_fallback_plan


def test_mask_from_request_role_bar():
    mask = mask_from_request(8, [3, 4], "bass")
    assert mask.mask.sum() == 2
    assert mask.mask[1, 2] == 1


def test_mask_from_constraints():
    plan = make_fallback_plan(4)
    for con in plan["constraints"]:
        con["editable"] = con["track_role"] == "lead" and con["bar_id"] == 2
    mask = mask_from_constraints(plan)
    assert mask.mask.sum() == 1
    assert mask.mask[0, 1] == 1
