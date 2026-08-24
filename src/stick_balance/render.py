from __future__ import annotations

import math
import os

import pygame

from stick_balance.physics import CartPoleEnv, PoleState

WIDTH = 900
HEIGHT = 480
SCALE = 140
CART_W = 80
CART_H = 28
POLE_PX = 180

BG = (10, 16, 22)
TRACK = (48, 72, 88)
CART = (90, 196, 186)
POLE = (232, 92, 78)
TEXT = (230, 234, 238)
MUTED = (140, 152, 164)
FAIL = (232, 92, 78)


def init_display(headless: bool) -> pygame.Surface:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.font.init()
    flags = 0
    return pygame.display.set_mode((WIDTH, HEIGHT), flags)


def draw(surface: pygame.Surface, env: CartPoleEnv, score: int, done: bool) -> None:
    surface.fill(BG)
    font = pygame.font.SysFont("DejaVu Sans", 20)
    small = pygame.font.SysFont("DejaVu Sans", 16)

    track_y = int(HEIGHT * 0.68)
    pygame.draw.line(surface, TRACK, (40, track_y), (WIDTH - 40, track_y), 3)

    state = env.state
    cart_x = WIDTH / 2 + state.x * SCALE
    cart_y = track_y
    cart_rect = pygame.Rect(0, 0, CART_W, CART_H)
    cart_rect.center = (int(cart_x), cart_y)
    pygame.draw.rect(surface, CART, cart_rect, border_radius=6)

    pole_x = cart_x + math.sin(state.theta) * POLE_PX
    pole_y = cart_y - math.cos(state.theta) * POLE_PX
    pygame.draw.line(
        surface,
        POLE,
        (int(cart_x), cart_y),
        (int(pole_x), int(pole_y)),
        8,
    )
    pygame.draw.circle(surface, POLE, (int(pole_x), int(pole_y)), 8)

    title = "Stick Balance — hold the pole upright"
    hint = "Left / Right arrows   |   R reset   |   Esc quit"
    status = "FALLEN — press R" if done else f"score {score}"
    color = FAIL if done else TEXT
    surface.blit(font.render(title, True, TEXT), (24, 18))
    surface.blit(small.render(hint, True, MUTED), (24, 48))
    surface.blit(font.render(status, True, color), (24, 78))
    deg = math.degrees(state.theta)
    surface.blit(small.render(f"angle {deg:+.1f}°   x {state.x:+.2f}", True, MUTED), (24, 112))


def snapshot(
    path: str,
    state: PoleState,
    score: int = 0,
    done: bool = False,
    headless: bool = True,
) -> str:
    surface = init_display(headless=headless)
    env = CartPoleEnv(seed=0)
    env.state = state
    draw(surface, env, score=score, done=done)
    pygame.image.save(surface, path)
    pygame.display.quit()
    return path
