from math import isfinite, radians
import random

from stick_balance.physics import Action, CartPoleEnv, PoleState


def test_reset_starts_near_upright():
    env = CartPoleEnv(seed=1)
    state = env.reset()
    assert isinstance(state, PoleState)
    assert abs(state.theta) < 0.1
    assert abs(state.x) < 0.1


def test_right_force_moves_cart_right():
    env = CartPoleEnv(seed=0)
    env.reset()
    env.state = PoleState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    for _ in range(20):
        env.step(Action.RIGHT)
    assert env.state.x > 0.05


def test_left_force_moves_cart_left():
    env = CartPoleEnv(seed=0)
    env.reset()
    env.state = PoleState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    for _ in range(20):
        env.step(Action.LEFT)
    assert env.state.x < -0.05


def test_pole_can_fall_until_horizontal():
    env = CartPoleEnv(seed=0)
    env.reset()
    env.state = PoleState(x=0.0, x_dot=0.0, theta=radians(89), theta_dot=0.0)
    _, _, done = env.step(Action.NONE)
    assert done is False

    env.state = PoleState(x=0.0, x_dot=0.0, theta=radians(90), theta_dot=0.0)
    _, _, done = env.step(Action.NONE)
    assert done is True


def test_cart_position_does_not_end_run_before_pole_is_horizontal():
    env = CartPoleEnv(seed=0)
    env.state = PoleState(x=env.x_limit + 1.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    _, _, done = env.step(Action.NONE)
    assert done is False


def test_cart_is_clamped_at_both_track_boundaries():
    for action, expected_x in (
        (Action.LEFT, -CartPoleEnv.x_limit),
        (Action.RIGHT, CartPoleEnv.x_limit),
    ):
        env = CartPoleEnv(seed=0)
        env.state = PoleState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
        for _ in range(500):
            env.step(action)
        assert env.state.x == expected_x
        assert env.state.x_dot == 0.0


def test_cart_drag_slows_coasting_after_input_is_released():
    env = CartPoleEnv(seed=0)
    env.state = PoleState(x=0.0, x_dot=2.0, theta=0.0, theta_dot=0.0)
    initial_speed = abs(env.state.x_dot)
    for _ in range(100):
        env.step(Action.NONE)
    assert abs(env.state.x_dot) < initial_speed * 0.1


def test_physics_keeps_advancing_with_no_input_after_fall():
    env = CartPoleEnv(seed=0)
    env.state = PoleState(x=0.0, x_dot=0.0, theta=radians(90), theta_dot=1.0)
    before = env.state
    after, _, done = env.step(Action.NONE)
    assert done is True
    assert after != before


def test_pole_drag_settles_a_fallen_pole_instead_of_spinning_forever():
    env = CartPoleEnv(seed=0)
    env.state = PoleState(x=0.0, x_dot=0.0, theta=radians(90), theta_dot=8.0)
    for _ in range(1500):
        env.step(Action.NONE)
    assert abs(env.state.theta_dot) < 0.01


def test_upright_idle_is_not_immediately_done():
    env = CartPoleEnv(seed=0)
    env.reset()
    env.state = PoleState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    _, reward, done = env.step(Action.NONE)
    assert done is False
    assert reward > 0


def test_step_is_deterministic_from_same_state():
    a = CartPoleEnv(seed=0)
    b = CartPoleEnv(seed=0)
    start = PoleState(x=0.01, x_dot=-0.02, theta=0.03, theta_dot=0.04)
    a.reset()
    b.reset()
    a.state = start
    b.state = start
    sa, ra, da = a.step(Action.RIGHT)
    sb, rb, db = b.step(Action.RIGHT)
    assert sa == sb
    assert ra == rb
    assert da is db


def test_long_random_run_stays_finite_and_inside_track():
    rng = random.Random(42)
    env = CartPoleEnv(seed=42)
    env.reset()
    for _ in range(10_000):
        state, _, _ = env.step(rng.choice(list(Action)))
        assert -env.x_limit <= state.x <= env.x_limit
        assert all(isfinite(value) for value in (
            state.x,
            state.x_dot,
            state.theta,
            state.theta_dot,
        ))
