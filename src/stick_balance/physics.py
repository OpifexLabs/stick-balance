from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import atan2, cos, pi, sin
import random


class Action(IntEnum):
    LEFT = -1
    NONE = 0
    RIGHT = 1


@dataclass(frozen=True, slots=True)
class PoleState:
    x: float
    x_dot: float
    theta: float
    theta_dot: float


class CartPoleEnv:
    """Classic inverted pendulum on a cart. theta=0 is upright."""

    gravity = 9.8
    cart_mass = 1.0
    pole_mass = 0.1
    pole_length = 0.5
    force_mag = 10.0
    dt = 0.02
    # Keep the cart and full pole inside the rendered track at every angle.
    x_limit = 1.6
    cart_drag = 1.2
    pole_drag = 0.8
    # A run is lost only once the pole reaches horizontal (90 degrees).
    theta_limit = pi / 2

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.state = PoleState(0.0, 0.0, 0.0, 0.0)

    def reset(self) -> PoleState:
        self.state = PoleState(
            x=self._rng.uniform(-0.05, 0.05),
            x_dot=self._rng.uniform(-0.05, 0.05),
            theta=self._rng.uniform(-0.05, 0.05),
            theta_dot=self._rng.uniform(-0.05, 0.05),
        )
        return self.state

    def step(self, action: Action | int) -> tuple[PoleState, float, bool]:
        force = self.force_mag * int(action)
        x, x_dot, theta, theta_dot = (
            self.state.x,
            self.state.x_dot,
            self.state.theta,
            self.state.theta_dot,
        )
        total_mass = self.cart_mass + self.pole_mass
        polemass_length = self.pole_mass * self.pole_length
        costheta = cos(theta)
        sintheta = sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_length * (4.0 / 3.0 - self.pole_mass * costheta * costheta / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Viscous drag dissipates energy instead of allowing endless coasting
        # or rotation. Semi-implicit Euler is substantially more stable for
        # this damped mechanical system than updating position first.
        xacc -= self.cart_drag * x_dot
        thetaacc -= self.pole_drag * theta_dot
        x_dot = x_dot + self.dt * xacc
        theta_dot = theta_dot + self.dt * thetaacc
        x = x + self.dt * x_dot
        theta = theta + self.dt * theta_dot

        # The visible track is a hard, inelastic stop. Cancel only velocity
        # pointing through a wall so the cart can immediately move away again.
        if x >= self.x_limit:
            x = self.x_limit
            x_dot = min(x_dot, 0.0)
        elif x <= -self.x_limit:
            x = -self.x_limit
            x_dot = max(x_dot, 0.0)

        # Keep angles numerically bounded while preserving the same direction.
        theta = atan2(sin(theta), cos(theta))
        self.state = PoleState(x, x_dot, theta, theta_dot)

        done = abs(theta) >= self.theta_limit
        reward = 0.0 if done else 1.0
        return self.state, reward, done
