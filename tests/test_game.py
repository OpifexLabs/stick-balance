from math import radians

from stick_balance.game import advance_frame
from stick_balance.physics import Action, CartPoleEnv, PoleState


def test_fallen_run_keeps_simulating_but_ignores_player_input():
    start = PoleState(x=0.0, x_dot=0.0, theta=radians(90), theta_dot=1.0)
    left = CartPoleEnv(seed=0)
    right = CartPoleEnv(seed=0)
    left.state = start
    right.state = start

    left_score, left_done = advance_frame(left, Action.LEFT, done=True)
    right_score, right_done = advance_frame(right, Action.RIGHT, done=True)

    assert left_done is True
    assert right_done is True
    assert left_score == right_score == 0
    assert left.state == right.state
    assert left.state != start
