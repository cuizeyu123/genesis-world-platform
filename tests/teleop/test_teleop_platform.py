import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "manipulation"))
from teleop_validation import validate_agent_plan, validate_move_delta


def test_agent_plan_accepts_a_bounded_move():
    delta = validate_agent_plan("move", [0.01, 0.0, -0.005])

    assert delta.tolist() == [0.01, 0.0, -0.005]


@pytest.mark.parametrize(
    "delta",
    [
        [0.031, 0.0, 0.0],  # Per-axis limit is 3 cm.
        [0.03, 0.03, 0.0],  # Combined movement exceeds 4 cm.
        [float("nan"), 0.0, 0.0],
    ],
)
def test_agent_plan_rejects_unsafe_or_invalid_moves(delta):
    with pytest.raises(ValueError):
        validate_agent_plan("move", delta)


def test_move_delta_rejects_wrong_shape():
    with pytest.raises(ValueError, match="格式无效"):
        validate_move_delta([0.01, 0.0])


@pytest.mark.parametrize("delta", [[0.0, 0.0], [0.0, float("inf"), 0.0]])
def test_non_move_plan_requires_empty_or_finite_xyz_delta(delta):
    with pytest.raises(ValueError):
        validate_agent_plan("observe", delta)


def test_non_move_plan_cannot_request_translation():
    with pytest.raises(ValueError, match="非移动动作"):
        validate_agent_plan("close_gripper", [0.01, 0.0, 0.0])


def test_observe_plan_accepts_empty_displacement():
    assert validate_agent_plan("observe", []).size == 0


def test_non_move_plan_accepts_zero_xyz_delta():
    assert validate_agent_plan("observe", [0.0, 0.0, 0.0]).tolist() == [0.0, 0.0, 0.0]
