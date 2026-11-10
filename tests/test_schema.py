from vhcmusic.planning.schema import PlanValidationError, make_fallback_plan, validate_plan


def test_fallback_plan_valid():
    plan = make_fallback_plan(8)
    summary = validate_plan(plan)
    assert summary.bars == 8
    assert len(plan["constraints"]) == 8 * 6


def test_invalid_range_fails():
    plan = make_fallback_plan(4)
    plan["structure"][0]["energy"] = 1.5
    try:
        validate_plan(plan)
    except PlanValidationError:
        return
    raise AssertionError("invalid energy should fail")
