import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import datetime
from pathlib import Path
from discord_webhook import DiscordWebhook, DiscordEmbed

import numpy as np

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
from tqdm import tqdm
from visualizer import initialize_pygame, visualize_game, display_training_done
import pygame
from utils import send_discord_file

visualize = True
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1329514649800151121/jcFewfurS-xkT2_E6PMnJhO7OFUmPiJU9SvcSjzTz6cbABqv9LQVz8H6VvAeBVBnslxu"

# Initialize Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

# Limit the action-space to
#   0. walk right
#   1. jump right
# env = JoypadSpace(env, [["right"], ["right", "A"], ["left"], ["left", "A"]])
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# Apply Wrappers to environment
env = SkipFrame(env, skip=2)

# rbg_display is used to display the game
rbg_display = env

env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.0)
env = FrameStack(env, num_stack=2)
env.reset()

# Create a separate environment for display
rbg_display = FrameStack(rbg_display, num_stack=4)
rbg_display.reset()


save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)


window_width, window_height = 256 * 2, 240 * 2 * 2
game_width, game_height = 256 * 2, 240 * 2

# Calculate the offset to center the game display
offset_x = (window_width - game_width) // 2
offset_y = (window_height - game_height) // 2

checkpoint = Path("checkpoints/2025-01-17T02-05-24/mario_net_22.chkpt")
# checkpoint = None

mario = Mario(
    state_dim=(2, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint,
)


# mario.curr_episode = 140
# mario.curr_episode = 39
# mario.reset_max_all()

logger = MetricLogger(save_dir)

episodes = 100000
worlds = range(1, 9)
stages = range(1, 5)

if visualize:
    (
        screen,
        clock,
        font,
        BUTTON_LIST,
        buttons,
        small_font,
        normal_font,
        BUTTON_LAYOUT,
    ) = initialize_pygame(window_width, window_height)


### for Loop that train the model num_episodes times by playing the game

with tqdm(total=episodes, initial=mario.curr_episode) as pbar:
    for e in range(mario.curr_episode, episodes):
        mario.curr_episode = e

        state = env.reset()
        rbg_state = rbg_display.reset()

        last_x_pos_screen = 0
        if visualize:
            last_x_pos_screen_time = pygame.time.get_ticks()
            # Play the game!
            # last_frame_time = pygame.time.get_ticks()
        while True:

            # 4. Run agent on the state
            action = mario.act(state)

            # 5. Agent performs action
            next_state, reward, done, info = env.step(action)

            mario.update_max(info)
            current_max = mario.get_max(info["world"], info["stage"])

            if visualize:
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

            # 6. Remember
            mario.cache(state, next_state, action, reward, done)

            # 7. Learn
            q, loss = mario.learn()

            # 8. Logging
            logger.log_step(reward, loss, q)

            # 9. Update state
            state = next_state

            # 10. Check if x_pos_screen has not changed in 15 seconds
            # debug
            # print(info["x_pos_screen"], last_x_pos_screen)
            # print(pygame.time.get_ticks(), last_x_pos_screen_time)
            if info["x_pos_screen"] != last_x_pos_screen:
                last_x_pos_screen = info["x_pos_screen"]
                last_x_pos_screen_time = pygame.time.get_ticks()
            elif pygame.time.get_ticks() - last_x_pos_screen_time > 10000:
                done = True

            # 11. Check if end of game
            # if done or info["flag_get"]:
            if info["flag_get"]:
                mario.update_max(info, win=True)
            if done and not info["flag_get"]:
                break

        logger.log_episode()
        pbar.set_postfix(
            {
                "step": mario.curr_step,
                "ep_reward": logger.ep_rewards[-1],
                "ep_length": logger.ep_lengths[-1],
                "ep_avg_losses": logger.ep_avg_losses[-1],
                "ep_avg_qs": logger.ep_avg_qs[-1],
                "x_pos": info["x_pos"],
                "max_x_pos": current_max["x_pos"],
            }
        )

        webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL)
        embed = DiscordEmbed(
            title=f"Episode {e} finished!",
            description="Mario finished an episode",
            color="03b2f8",
        )

        embed.add_embed_field(name="Episode", value=str(e))
        embed.add_embed_field(name="Step", value=str(mario.curr_step))
        embed.add_embed_field(name="Reward", value=str(logger.ep_rewards[-1]))
        embed.add_embed_field(name="Length", value=str(logger.ep_lengths[-1]))
        embed.add_embed_field(name="Avg Losses", value=str(logger.ep_avg_losses[-1]))
        embed.add_embed_field(name="Avg Qs", value=str(logger.ep_avg_qs[-1]))
        embed.add_embed_field(name="X Pos", value=str(info["x_pos"]))
        embed.add_embed_field(name="Max X Pos", value=str(current_max["x_pos"]))
        webhook.add_embed(embed)
        webhook.execute()

        if e % 20 == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )
            send_discord_file(
                logger.ep_rewards_plot,
                DISCORD_WEBHOOK_URL,
                "Episode Rewards",
                f"current: {logger.ep_rewards[-1]}",
            )
            send_discord_file(
                logger.ep_lengths_plot,
                DISCORD_WEBHOOK_URL,
                "Episode Lengths",
                f"current: {logger.ep_lengths[-1]}",
            )
            send_discord_file(
                logger.ep_avg_losses_plot,
                DISCORD_WEBHOOK_URL,
                "Episode Avg Losses",
                f"current: {logger.ep_avg_losses[-1]}",
            )
            send_discord_file(
                logger.ep_avg_qs_plot,
                DISCORD_WEBHOOK_URL,
                "Episode Avg Qs",
                f"current: {logger.ep_avg_qs[-1]}",
            )

        pbar.update()


# after the training is done, display the notification on the pygame window
if visualize:
    display_training_done(screen, clock, normal_font, window_width, window_height)
