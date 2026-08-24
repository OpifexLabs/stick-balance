from __future__ import annotations

from dataclasses import dataclass

from stick_balance.physics import Action


@dataclass(frozen=True, slots=True)
class Control:
    mode: str  # pid | action | lock
    action: Action = Action.NONE
    override: bool = False


def select_control(*, auto_pid: bool, done: bool, left: bool, right: bool) -> Control:
    """PID holds until arrows override; releasing arrows resumes PID."""
    if done:
        return Control(mode="lock")
    if left and not right:
        return Control(mode="action", action=Action.LEFT, override=auto_pid)
    if right and not left:
        return Control(mode="action", action=Action.RIGHT, override=auto_pid)
    if auto_pid:
        return Control(mode="pid")
    return Control(mode="action", action=Action.NONE)
