from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import sqrt
from statistics import mean
from typing import Callable

from stick_balance.physics import Action, CartPoleEnv, PoleState

ForceFn = Callable[[PoleState, float], float]


@dataclass(frozen=True, slots=True)
class PidGains:
    kp_theta: float = 0.0
    ki_theta: float = 0.0
    kd_theta: float = 0.0
    kp_x: float = 0.0
    ki_x: float = 0.0
    kd_x: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# Frozen from logged search: iteration 80 of 100 in logs/pid_tune.csv
BEST_GAINS = PidGains(
    kp_theta=61.429406,
    ki_theta=0.549106,
    kd_theta=18.813061,
    kp_x=2.8,
    ki_x=0.020416,
    kd_x=1.896718,
)


class StatePid:
    """Full-state PID: angle + cart position, with integrator anti-windup."""

    def __init__(self, gains: PidGains, force_limit: float = 10.0) -> None:
        self.gains = gains
        self.force_limit = force_limit
        self.int_theta = 0.0
        self.int_x = 0.0

    def reset(self) -> None:
        self.int_theta = 0.0
        self.int_x = 0.0

    def force(self, state: PoleState, dt: float = CartPoleEnv.dt) -> float:
        g = self.gains
        raw = (
            g.kp_theta * state.theta
            + g.ki_theta * self.int_theta
            + g.kd_theta * state.theta_dot
            + g.kp_x * state.x
            + g.ki_x * self.int_x
            + g.kd_x * state.x_dot
        )
        limited = max(-self.force_limit, min(self.force_limit, raw))
        unsaturated = abs(raw) <= self.force_limit
        if unsaturated or raw * state.theta <= 0:
            self.int_theta += state.theta * dt
        if unsaturated or raw * state.x <= 0:
            self.int_x += state.x * dt
        self.int_theta = max(-2.0, min(2.0, self.int_theta))
        self.int_x = max(-2.0, min(2.0, self.int_x))
        return limited


def force_to_action(force: float, deadband: float = 0.5) -> Action:
    if force > deadband:
        return Action.RIGHT
    if force < -deadband:
        return Action.LEFT
    return Action.NONE


@dataclass(slots=True)
class EvalMetrics:
    cost: float
    survived_frac: float
    mean_abs_x: float
    mean_abs_theta: float
    mean_abs_x_dot: float
    mean_abs_theta_dot: float
    mean_abs_u: float
    settle_abs_x: float
    settle_abs_theta: float


def evaluate_controller(
    force_fn: ForceFn,
    *,
    horizon: int = 1000,
    starts: list[PoleState] | None = None,
    dt: float = CartPoleEnv.dt,
) -> EvalMetrics:
    if starts is None:
        starts = default_starts()

    survived = 0
    abs_x: list[float] = []
    abs_th: list[float] = []
    abs_xd: list[float] = []
    abs_thd: list[float] = []
    abs_u: list[float] = []
    settle_x: list[float] = []
    settle_th: list[float] = []
    settle_from = horizon // 2

    for start in starts:
        env = CartPoleEnv(seed=0)
        env.state = start
        fell = False
        for t in range(horizon):
            u = float(force_fn(env.state, dt))
            state, _, done = env.step_force(u)
            abs_x.append(abs(state.x))
            abs_th.append(abs(state.theta))
            abs_xd.append(abs(state.x_dot))
            abs_thd.append(abs(state.theta_dot))
            abs_u.append(abs(u))
            if t >= settle_from:
                settle_x.append(abs(state.x))
                settle_th.append(abs(state.theta))
            if done:
                fell = True
                break
        if not fell:
            survived += 1

    n = max(len(starts), 1)
    survived_frac = survived / n
    fall_penalty = (1.0 - survived_frac) * 1_000_000.0
    # Stillness dominates: late-horizon pose and velocity, then effort.
    cost = (
        fall_penalty
        + 80.0 * _rms(settle_x)
        + 400.0 * _rms(settle_th)
        + 20.0 * _rms(abs_xd)
        + 80.0 * _rms(abs_thd)
        + 0.4 * _rms(abs_u)
        + 8.0 * _rms(abs_x)
        + 40.0 * _rms(abs_th)
    )
    return EvalMetrics(
        cost=cost,
        survived_frac=survived_frac,
        mean_abs_x=_mean(abs_x),
        mean_abs_theta=_mean(abs_th),
        mean_abs_x_dot=_mean(abs_xd),
        mean_abs_theta_dot=_mean(abs_thd),
        mean_abs_u=_mean(abs_u),
        settle_abs_x=_mean(settle_x),
        settle_abs_theta=_mean(settle_th),
    )


def default_starts() -> list[PoleState]:
    return [
        PoleState(0.0, 0.0, 0.0, 0.0),
        PoleState(0.0, 0.0, 0.03, 0.0),
        PoleState(0.0, 0.0, -0.03, 0.0),
        PoleState(0.15, 0.0, 0.02, 0.0),
        PoleState(-0.15, 0.0, -0.02, 0.0),
        PoleState(0.0, 0.15, 0.01, 0.0),
        PoleState(0.0, -0.15, -0.01, 0.0),
        PoleState(0.05, 0.05, 0.04, 0.05),
        PoleState(-0.08, -0.04, -0.035, -0.03),
        PoleState(0.0, 0.0, 0.06, 0.0),
        PoleState(0.25, 0.0, 0.015, 0.0),
        PoleState(-0.25, 0.0, -0.015, 0.0),
    ]


def evaluate_gains(gains: PidGains, **kwargs) -> EvalMetrics:
    # Fresh PID per start so integrators do not leak across scenarios.
    starts = kwargs.pop("starts", None) or default_starts()
    horizon = kwargs.get("horizon", 1000)
    dt = kwargs.get("dt", CartPoleEnv.dt)

    def run(start: PoleState) -> EvalMetrics:
        pid = StatePid(gains)
        return evaluate_controller(pid.force, horizon=horizon, starts=[start], dt=dt)

    parts = [run(start) for start in starts]
    survived = mean(p.survived_frac for p in parts)
    return EvalMetrics(
        cost=mean(p.cost for p in parts),
        survived_frac=survived,
        mean_abs_x=mean(p.mean_abs_x for p in parts),
        mean_abs_theta=mean(p.mean_abs_theta for p in parts),
        mean_abs_x_dot=mean(p.mean_abs_x_dot for p in parts),
        mean_abs_theta_dot=mean(p.mean_abs_theta_dot for p in parts),
        mean_abs_u=mean(p.mean_abs_u for p in parts),
        settle_abs_x=mean(p.settle_abs_x for p in parts),
        settle_abs_theta=mean(p.settle_abs_theta for p in parts),
    )


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return sqrt(sum(v * v for v in values) / len(values))


GAIN_NAMES = tuple(f.name for f in fields(PidGains))
