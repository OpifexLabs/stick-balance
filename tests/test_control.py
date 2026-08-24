from stick_balance.physics import Action
from stick_balance.control import select_control


def test_pid_mode_without_keys_uses_pid():
    cmd = select_control(auto_pid=True, done=False, left=False, right=False)
    assert cmd.mode == "pid"
    assert cmd.override is False


def test_arrows_override_pid_and_resume_when_released():
    left = select_control(auto_pid=True, done=False, left=True, right=False)
    assert left.mode == "action"
    assert left.action == Action.LEFT
    assert left.override is True

    right = select_control(auto_pid=True, done=False, left=False, right=True)
    assert right.action == Action.RIGHT
    assert right.override is True

    resume = select_control(auto_pid=True, done=False, left=False, right=False)
    assert resume.mode == "pid"
    assert resume.override is False


def test_fallen_locks_both_pid_and_arrows():
    cmd = select_control(auto_pid=True, done=True, left=True, right=False)
    assert cmd.mode == "lock"
