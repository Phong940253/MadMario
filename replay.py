import random, datetime
from pathlib import Path

import pygame
import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

env = JoypadSpace(env, COMPLEX_MOVEMENT)

env = SkipFrame(env, skip=2)
# rbg_display is used to display the game
rbg_display = env

env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.0)
env = FrameStack(env, num_stack=2)
env.reset()

# Create a separate environment for display
rbg_display = FrameStack(rbg_display, num_stack=2)
rbg_display.reset()

pygame.init()

window_width, window_height = 256 * 2, 240 * 2 * 2
game_width, game_height = 256 * 2, 240 * 2

# Calculate the offset to center the game display
offset_x = (window_width - game_width) // 2
offset_y = (window_height - game_height) // 2

screen = pygame.display.set_mode((window_width, window_height))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)
fps_text = None
fps_history_100 = []
fps_history_1000 = []


save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

checkpoint = Path("checkpoints/2025-01-17T08-13-50/mario_net_40.chkpt")
mario = Mario(
    state_dim=(2, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint,
)
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()

    rbg_state = rbg_display.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        # 3. Show environment (the visual)
        # from (2, 240, 256, 3) to (240, 256, 3)
        for frame in rbg_state:
            frame = np.transpose(frame, (1, 0, 2))
            surface = pygame.surfarray.make_surface(frame)
            scaled_surface = pygame.transform.scale(surface, (game_width, game_height))

            # Center the game display in the window
            screen.fill((0, 0, 0))  # Clear the screen with a black background
            screen.blit(scaled_surface, (offset_x, offset_y))

            fps = clock.get_fps()
            fps_history_100.append(fps)
            fps_history_1000.append(fps)
            if len(fps_history_100) > 100:
                fps_history_100.pop(0)
            if len(fps_history_1000) > 1000:
                fps_history_1000.pop(0)

            fps_1_percent = (
                sorted(fps_history_100)[int(len(fps_history_100) * 0.01)]
                if len(fps_history_100) > 0
                else 0
            )
            fps_01_percent = (
                sorted(fps_history_1000)[int(len(fps_history_1000) * 0.001)]
                if len(fps_history_1000) > 0
                else 0
            )

            fps_text = font.render(
                f"FPS: {fps:.2f}  1%: {fps_1_percent:.2f}  0.1%: {fps_01_percent:.2f}",
                True,
                (255, 255, 255),
            )
            screen.blit(fps_text, (10, 10))
            pygame.display.update()
            clock.tick(60)

        action = mario.act(state)

        next_state, reward, done, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
