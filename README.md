# Stick Balance

First Opifex step on the path **simulation → classical control → RL → simplest physical robot**.

This repo is a playable inverted pendulum: a cart and a pole. Press **left / right** to keep the pole upright, or run the tuned PID so it stands almost still. Physics is a separate module so the same environment can later be driven by RL.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy pytest
PYTHONPATH=src python -m stick_balance.game --pid
```

Keys: `←` / `→` or `A` / `D` to push the cart. `P` toggles the logged PID. The cart and pole remain inside the window; horizontal drag prevents endless coasting. The run ends when the pole reaches 90° (horizontal). Physics keeps running with angular damping after a fall, but movement controls stay locked until `R` resets. `Esc` quits.

## PID

`BEST_GAINS` comes from 100 logged iterations in `logs/pid_tune.csv`. The search minimized late-horizon motion, not just survival. Re-run:

```bash
PYTHONPATH=src python -m stick_balance.tune_pid
```

Headless screenshots:

```bash
PYTHONPATH=src python -m stick_balance.screenshots
```

## Test

```bash
PYTHONPATH=src pytest -q
```

## Layout

- `src/stick_balance/physics.py` — cart-pole dynamics (`theta = 0` is upright)
- `src/stick_balance/game.py` — pygame loop
- `src/stick_balance/render.py` — drawing + PNG snapshots
- `src/stick_balance/pid.py` — state PID + frozen best gains
- `src/stick_balance/tune_pid.py` — 100-iteration logged search
- `logs/pid_tune.csv` — every iteration's cost and gains
- `assets/` — captured frames

No identity, no secrets, no hardware yet.
