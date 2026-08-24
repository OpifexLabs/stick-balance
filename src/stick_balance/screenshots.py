from __future__ import annotations

from math import radians
from pathlib import Path

from stick_balance.physics import Action, CartPoleEnv, PoleState
from stick_balance.render import snapshot


def main() -> list[Path]:
    out = Path(__file__).resolve().parents[2] / "assets"
    out.mkdir(parents=True, exist_ok=True)

    start = PoleState(x=0.0, x_dot=0.0, theta=0.08, theta_dot=0.0)
    p1 = out / "01-start.png"
    snapshot(str(p1), start, score=0, done=False)

    env = CartPoleEnv(seed=0)
    env.state = start
    score = 0
    done = False
    for i in range(18):
        action = Action.RIGHT if env.state.theta > 0 else Action.LEFT
        _, reward, done = env.step(action)
        score += int(reward)
        if done:
            break
    p2 = out / "02-correcting.png"
    snapshot(str(p2), env.state, score=score, done=done)

    env.state = PoleState(
        x=env.x_limit,
        x_dot=0.0,
        theta=radians(90),
        theta_dot=1.2,
    )
    p3 = out / "03-fallen.png"
    snapshot(str(p3), env.state, score=score, done=True)

    return [p1, p2, p3]


if __name__ == "__main__":
    for path in main():
        print(path)
