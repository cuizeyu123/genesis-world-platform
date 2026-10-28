"""Pure validation helpers for teleoperation actions, independent of Genesis runtime."""

import numpy as np


AGENT_MOVE_MAX_M = 0.03
AGENT_MOVE_MAX_NORM_M = 0.04


def validate_move_delta(delta_m):
    """Return a finite XYZ displacement that satisfies the VLM single-step limits."""
    try:
        delta = np.asarray(delta_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("VLM 移动量格式无效") from exc
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("VLM 移动量格式无效")
    if np.max(np.abs(delta)) > AGENT_MOVE_MAX_M + 1e-9 or np.linalg.norm(delta) > AGENT_MOVE_MAX_NORM_M:
        raise ValueError("VLM 移动量超过单步安全范围，请重新分析")
    return delta


def validate_agent_plan(action, delta_m):
    """Validate a structured plan's displacement before it is offered or queued."""
    try:
        delta = np.asarray(delta_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("VLM 位移建议包含无效数值，请重新分析") from exc

    if delta.size:
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            if action == "move":
                raise ValueError("VLM 移动建议格式无效，请重新分析")
            raise ValueError("VLM 位移建议包含无效数值，请重新分析")

    if action == "move":
        try:
            return validate_move_delta(delta)
        except ValueError as exc:
            if "格式无效" in str(exc):
                raise ValueError("VLM 移动建议格式无效，请重新分析") from exc
            raise ValueError("VLM 给出的移动建议超出单步限幅，请重新分析") from exc

    if delta.size and np.any(np.abs(delta) > 1e-9):
        raise ValueError("非移动动作不应包含末端位移")
    return delta
