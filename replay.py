import datetime
from pathlib import Path

import pygame
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
from visualizer import initialize_pygame, visualize_game, display_training_done

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

window_width, window_height = 256 * 2, 240 * 2 * 2
game_width, game_height = 256 * 2, 240 * 2

# Calculate the offset to center the game display
offset_x = (window_width - game_width) // 2
offset_y = (window_height - game_height) // 2

screen, clock, font, BUTTON_LIST, buttons, small_font, normal_font, BUTTON_LAYOUT = (
    initialize_pygame(window_width, window_height)
)

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
current_max = {"x_pos": 0, "score": 0, "coins": 0, "time_left": 0}

for e in range(episodes):

    state = env.reset()

    rbg_state = rbg_display.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = mario.act(state)

        next_state, reward, done, info = env.step(action)

        mario.update_max(info)
        current_max = mario.get_max(info["world"], info["stage"])
        # 3. Show environment (the visual)
        # from (2, 240, 256, 3) to (240, 256, 3)

        visualize_game(
            screen,
            clock,
            font,
            BUTTON_LIST,
            buttons,
            small_font,
            normal_font,
            BUTTON_LAYOUT,
            rbg_state,
            game_width,
            game_height,
            offset_x,
            offset_y,
            action,
            info,
            e,
            mario,
            logger,
            current_max,
            window_width,
            window_height,
        )

        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if info["flag_get"]:
            mario.update_max(info, win=True)

        if done and not info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

display_training_done(screen, clock, normal_font, window_width, window_height)
