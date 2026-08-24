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


def test_large_angle_is_fallen():
    env = CartPoleEnv(seed=0)
    env.reset()
    env.state = PoleState(x=0.0, x_dot=0.0, theta=0.5, theta_dot=0.0)
    _, _, done = env.step(Action.NONE)
    assert done is True


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
