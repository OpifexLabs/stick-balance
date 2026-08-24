# Stick Balance

First Opifex step on the path **simulation → classical control → RL → simplest physical robot**.

This repo is a playable inverted pendulum: a cart and a pole. Press **left / right** to keep the pole upright. Physics is a separate module so the same environment can later be driven by PID or RL.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy pytest
PYTHONPATH=src python -m stick_balance.game
```

Keys: `←` / `→` or `A` / `D` to push the cart. The cart and pole remain inside the window; horizontal drag prevents endless coasting. The run ends when the pole reaches 90° (horizontal). Physics keeps running with angular damping after a fall, but movement controls stay locked until `R` resets. `Esc` quits.

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
- `assets/` — captured frames

No identity, no secrets, no hardware yet.
