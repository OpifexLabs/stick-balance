from __future__ import annotations

import sys

import pygame

from stick_balance.physics import Action, CartPoleEnv
from stick_balance.render import draw, init_display


def advance_frame(env: CartPoleEnv, action: Action, done: bool) -> tuple[int, bool]:
    """Advance one frame; a fallen run keeps simulating with controls locked."""
    if done:
        env.step(Action.NONE)
        return 0, True

    _, reward, fell = env.step(action)
    return int(reward), fell


def main() -> None:
    headless = "--headless" in sys.argv
    screen = init_display(headless=headless)
    pygame.display.set_caption("Stick Balance")
    clock = pygame.time.Clock()
    env = CartPoleEnv()
    env.reset()
    score = 0
    done = False

    running = True
    while running:
        action = Action.NONE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    score = 0
                    done = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = Action.LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = Action.RIGHT

        gained, done = advance_frame(env, action, done)
        score += gained

        draw(screen, env, score=score, done=done)
        pygame.display.flip()
        clock.tick(50)

    pygame.quit()


if __name__ == "__main__":
    main()
