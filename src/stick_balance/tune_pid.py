"""100-iteration logged PID search for a near-still cart-pole hold.

Each iteration is scored on a fixed start suite. Cost heavily punishes
falling and residual late-horizon motion. Results are appended to
logs/pid_tune.csv so later runs can be compared.
"""

from __future__ import annotations

import csv
import random
from dataclasses import replace
from pathlib import Path

from stick_balance.pid import (
    GAIN_NAMES,
    BEST_GAINS,
    EvalMetrics,
    PidGains,
    evaluate_gains,
)

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "pid_tune.csv"
ITERATIONS = 100

# Physically motivated seed: stiff angle PD, milder cart PD, tiny I.
SEED = PidGains(
    kp_theta=42.0,
    ki_theta=0.25,
    kd_theta=9.0,
    kp_x=2.8,
    ki_x=0.04,
    kd_x=3.2,
)

# Relative perturbation scale per gain, shrunk over time.
SCALES = {
    "kp_theta": 12.0,
    "ki_theta": 0.25,
    "kd_theta": 3.0,
    "kp_x": 1.4,
    "ki_x": 0.04,
    "kd_x": 1.2,
}


def _row(iteration: int, gains: PidGains, metrics: EvalMetrics, best_cost: float) -> dict:
    data = {
        "iteration": iteration,
        "cost": f"{metrics.cost:.6f}",
        "best_cost": f"{best_cost:.6f}",
        "survived_frac": f"{metrics.survived_frac:.4f}",
        "mean_abs_x": f"{metrics.mean_abs_x:.6f}",
        "mean_abs_theta": f"{metrics.mean_abs_theta:.6f}",
        "mean_abs_x_dot": f"{metrics.mean_abs_x_dot:.6f}",
        "mean_abs_theta_dot": f"{metrics.mean_abs_theta_dot:.6f}",
        "mean_abs_u": f"{metrics.mean_abs_u:.6f}",
        "settle_abs_x": f"{metrics.settle_abs_x:.6f}",
        "settle_abs_theta": f"{metrics.settle_abs_theta:.6f}",
    }
    data.update({name: f"{getattr(gains, name):.6f}" for name in GAIN_NAMES})
    return data


def propose(best: PidGains, iteration: int, rng: random.Random) -> PidGains:
    """Coordinate descent with shrinking noise, plus occasional joint jitter."""
    shrink = 0.85 ** (iteration // 8)
    values = best.as_dict()
    if iteration == 0:
        return SEED
    if iteration % 7 == 0:
        for name in GAIN_NAMES:
            values[name] += rng.gauss(0.0, SCALES[name] * shrink)
    else:
        name = GAIN_NAMES[(iteration - 1) % len(GAIN_NAMES)]
        step = SCALES[name] * shrink
        # Alternate signed steps then a gaussian probe.
        phase = (iteration - 1) // len(GAIN_NAMES)
        if phase % 3 == 0:
            values[name] += step
        elif phase % 3 == 1:
            values[name] -= step
        else:
            values[name] += rng.gauss(0.0, step)
    for name, value in values.items():
        values[name] = max(0.0, value)
    return PidGains(**values)


def run(
    iterations: int = ITERATIONS,
    seed: int = 7,
    *,
    start_iteration: int = 0,
    initial: PidGains | None = None,
    append: bool = False,
) -> tuple[PidGains, EvalMetrics, Path]:
    rng = random.Random(seed)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "cost",
        "best_cost",
        "survived_frac",
        "mean_abs_x",
        "mean_abs_theta",
        "mean_abs_x_dot",
        "mean_abs_theta_dot",
        "mean_abs_u",
        "settle_abs_x",
        "settle_abs_theta",
        *GAIN_NAMES,
    ]
    best = initial if initial is not None else SEED
    best_metrics = evaluate_gains(best)
    rows = []
    for offset in range(iterations):
        i = start_iteration + offset
        candidate = best if (append and offset == 0) else propose(best, i, rng)
        if append and offset == 0:
            candidate = propose(best, max(i, 1), rng)
        metrics = evaluate_gains(candidate)
        if metrics.cost < best_metrics.cost:
            best = candidate
            best_metrics = metrics
        rows.append(_row(i, candidate, metrics, best_metrics.cost))
        print(
            f"iter {i:03d} cost={metrics.cost:10.3f} "
            f"surv={metrics.survived_frac:.2f} "
            f"settle|θ|={metrics.settle_abs_theta:.5f} "
            f"settle|x|={metrics.settle_abs_x:.5f} "
            f"best={best_metrics.cost:.3f}",
            flush=True,
        )
    mode = "a" if append and LOG_PATH.exists() else "w"
    with LOG_PATH.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    return best, best_metrics, LOG_PATH


if __name__ == "__main__":
    import sys

    extra = "--continue" in sys.argv
    if extra:
        gains, metrics, path = run(
            iterations=200,
            seed=19,
            start_iteration=100,
            initial=BEST_GAINS,
            append=True,
        )
    else:
        gains, metrics, path = run()
    print("---")
    print("BEST", gains)
    print(
        f"cost={metrics.cost:.4f} survived={metrics.survived_frac:.3f} "
        f"settle|x|={metrics.settle_abs_x:.6f} settle|θ|={metrics.settle_abs_theta:.6f}"
    )
    print("log", path)
