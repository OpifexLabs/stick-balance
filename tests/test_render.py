from pathlib import Path

from stick_balance.physics import CartPoleEnv, PoleState
from stick_balance.render import POLE_PX, SCALE, WIDTH, snapshot


def test_snapshot_writes_png(tmp_path: Path):
    path = tmp_path / "frame.png"
    snapshot(
        str(path),
        PoleState(x=0.0, x_dot=0.0, theta=0.05, theta_dot=0.0),
        score=3,
        done=False,
    )
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_full_pole_stays_inside_window_at_cart_limits():
    furthest_endpoint = WIDTH / 2 + CartPoleEnv.x_limit * SCALE + POLE_PX
    assert furthest_endpoint <= WIDTH
