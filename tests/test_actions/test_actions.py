import numpy as np
import pytest

from twist.actions import (
    ActionPlan,
    ActionSpace,
    ActionValidationError,
    HighLevelControl,
    IntervalIntervention,
    ParametricPulse,
    PointIntervention,
    ScheduleIntervention,
    action_from_dict,
)
from twist.actions.costs import action_costs
from twist.actions.feasibility import (
    AmplitudeBounds,
    ForbiddenCombination,
    ImmutableVariables,
    InterventionBudget,
    MaxDuration,
    MaxRateOfChange,
    MinLeadTime,
    StateBounds,
    build_constraints,
    check_feasibility,
)
from twist.actions.trajectory_adapters import (
    first_change_atomic,
    first_changed_segment,
    full_changed_schedule,
)

T, K = 10, 3
ACTIONS = [
    PointIntervention(2, [0, 2], [1.0, -1.0]),
    ScheduleIntervention([(1, [0], [0.5]), (4, [1, 2], [1.0, 2.0])]),
    IntervalIntervention(3, 6, [1], [0.7]),
    IntervalIntervention(3, 5, [0, 1], [[1, 2], [3, 4]]),
    ParametricPulse(2, 4, [0], 0.5, "raised_cosine"),
    ParametricPulse(2, 3, [1, 2], [1.0, -1.0], "rect"),
    HighLevelControl("u_storage_dispatch", 24, 32, [0.15] * 8),
]


@pytest.mark.parametrize("action", ACTIONS, ids=lambda a: a.kind)
def test_serialization_roundtrip(action):
    d = action.to_dict()
    assert action_from_dict(d).to_dict() == d
    plan = ActionPlan(action, native=True, semantics="test", constraints=["ramp_limit"])
    assert ActionPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_point_schedule():
    s = PointIntervention(2, [0, 2], [1.0, -1.0]).to_schedule(T, K)
    assert s.mask.sum() == 2 and s.mask[2, 0] and s.mask[2, 2]
    assert s.value[2, 2] == -1.0 and s.n_timesteps == 1 and s.first_time == 2


def test_interval_and_pulse_schedules():
    s = IntervalIntervention(3, 5, [0, 1], [[1, 2], [3, 4]]).to_schedule(T, K)
    np.testing.assert_array_equal(s.value[3:5, :2], [[1, 2], [3, 4]])
    assert s.n_timesteps == 2
    p = ParametricPulse(2, 4, [0], 0.5).to_schedule(T, K)
    assert not p.mask.any() and p.n_timesteps == 4 and p.shift[2:6, 0].max() <= 0.5


def test_high_level_control_requires_world_interface():
    with pytest.raises(NotImplementedError):
        ACTIONS[-1].to_schedule(T, K)


def test_validation_reports_readable_violations():
    space = ActionSpace(T, K, ["a", "b", "c"], actionable=[0, 1], lower=[-1] * K, upper=[1] * K)
    PointIntervention(2, [0], [0.5]).check(space)
    with pytest.raises(ActionValidationError) as exc:
        PointIntervention(2, [2], [5.0]).check(space)
    msg = str(exc.value)
    assert "non-actionable" in msg and "'c'" in msg and "upper bound" in msg
    assert PointIntervention(T + 3, [0], [0.0]).validate(space)
    assert PointIntervention(1, [7], [0.0]).validate(space)


def test_costs_hand_case():
    x = np.zeros((T, K))
    s = ScheduleIntervention([(1, [0], [1.0]), (2, [0], [3.0])]).to_schedule(T, K)
    c = action_costs(s, x)
    assert c["duration"] == 2 and c["magnitude"] == 4.0
    assert c["smoothness"] == 1.0 + 4.0 + 9.0  # (1-0)^2 + (3-1)^2 + (0-3)^2


def test_feasibility_constraints():
    x = np.zeros((T, K))
    s = ScheduleIntervention([(1, [0], [1.0]), (8, [0, 1], [3.0, 1.0])]).to_schedule(T, K)
    rep = np.cumsum(np.ones((T, K)), axis=0)
    checks = [
        (ImmutableVariables([1]), False),
        (ImmutableVariables([2]), True),
        (AmplitudeBounds(2.0), False),
        (AmplitudeBounds(3.0), True),
        (StateBounds(0, 5), False),
        (MaxRateOfChange(1.0), True),
        (InterventionBudget(4.0), False),
        (MaxDuration(2), True),
        (MaxDuration(1), False),
        (MinLeadTime(t_label=9, min_lead=9), False),
        (ForbiddenCombination([0, 1]), False),
    ]
    for c, expected in checks:
        r = c.evaluate(x, s, rep)
        assert r.passed is expected, (c, r)
        assert r.passed or r.message
    report = check_feasibility(
        build_constraints([{"name": "max_duration", "max_timesteps": 1}]), x, s, rep
    )
    assert not report.passed and report.violations[0].name == "max_duration"


def test_trajectory_adapters_are_marked_non_native():
    x = np.zeros((T, K))
    xh = x.copy()
    xh[3, 1] = 1.0
    xh[4, 0] = 2.0
    xh[7, 2] = -1.0
    a = first_change_atomic(x, xh)
    assert not a.native and a.adapter == "first_change_atomic"
    assert a.action.to_dict() == {"kind": "point", "t": 3, "targets": [1], "values": [1.0]}
    seg = first_changed_segment(x, xh).action.to_schedule(T, K)
    assert seg.active_timesteps.tolist() == [3, 4]
    full = full_changed_schedule(x, xh).action.to_schedule(T, K)
    assert full.active_timesteps.tolist() == [3, 4, 7]
    assert first_change_atomic(x, x) is None and full_changed_schedule(x, x) is None


def test_bounds_tolerate_round_off_but_not_real_violations():
    x = np.full((T, K), 0.1)
    on_bound = x + 3.0  # (x + 3) − x is 3 ± ulp
    s = ScheduleIntervention([(2, [0], [on_bound[2, 0]])]).to_schedule(T, K)
    assert AmplitudeBounds(3.0).evaluate(x, s, None).passed
    s_over = ScheduleIntervention([(2, [0], [x[2, 0] + 3.0 + 1e-6])]).to_schedule(T, K)
    assert not AmplitudeBounds(3.0).evaluate(x, s_over, None).passed
