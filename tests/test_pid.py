from math import isfinite

from stick_balance.physics import CartPoleEnv, PoleState
from stick_balance.pid import PidGains, StatePid, evaluate_controller, force_to_action


def test_pid_pushes_right_when_pole_leans_right():
    pid = StatePid(PidGains(kp_theta=20.0, kd_theta=4.0, kp_x=1.0, kd_x=1.0))
    force = pid.force(PoleState(x=0.0, x_dot=0.0, theta=0.05, theta_dot=0.0))
    assert force > 0


def test_pid_pushes_left_when_pole_leans_left():
    pid = StatePid(PidGains(kp_theta=20.0, kd_theta=4.0, kp_x=1.0, kd_x=1.0))
    force = pid.force(PoleState(x=0.0, x_dot=0.0, theta=-0.05, theta_dot=0.0))
    assert force < 0


def test_force_to_action_uses_deadband():
    assert force_to_action(0.2, deadband=0.5) == 0
    assert force_to_action(6.0, deadband=0.5) == 1
    assert force_to_action(-6.0, deadband=0.5) == -1


def test_evaluate_controller_penalizes_a_fall_more_than_a_hold():
    env = CartPoleEnv(seed=0)
    env.state = PoleState(x=0.0, x_dot=0.0, theta=0.02, theta_dot=0.0)

    def hold(_state, _dt):
        return 0.0

    def shove(_state, _dt):
        return 10.0

    hold_metrics = evaluate_controller(hold, horizon=200, starts=[env.state])
    shove_metrics = evaluate_controller(shove, horizon=200, starts=[
        PoleState(x=0.0, x_dot=0.0, theta=0.02, theta_dot=0.0)
    ])
    assert hold_metrics.survived_frac >= shove_metrics.survived_frac
    assert isfinite(hold_metrics.cost)
    assert hold_metrics.cost < shove_metrics.cost or shove_metrics.survived_frac < 1.0
