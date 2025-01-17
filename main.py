import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import random, datetime
from pathlib import Path
from discord_webhook import DiscordWebhook, DiscordEmbed

import pygame
import numpy as np

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
from tqdm import tqdm

visualize = False
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

if visualize:
    pygame.init()

    screen = pygame.display.set_mode((256 * 2, 240 * 2 * 2))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    BUTTON_LIST = ["A", "B", "up", "down", "left", "right"]

    button_width = 40
    button_height = 20
    X_FROM_CENTER = 40
    Y_FROM_CENTER = 20
    X_CENTER = 60
    Y_CENTER = 330
    MARGIN_JOYPAD = 10
    MARGIN_BUTTON = 10

    # Draw buttons
    BUTTON_LAYOUT = {
        "A": (X_CENTER + 2 * button_width + MARGIN_JOYPAD, Y_CENTER),
        "B": (X_CENTER + 3 * button_width + MARGIN_JOYPAD + MARGIN_BUTTON, Y_CENTER),
        "up": (X_CENTER, Y_CENTER - Y_FROM_CENTER),
        "down": (X_CENTER, Y_CENTER + Y_FROM_CENTER),
        "left": (X_CENTER - X_FROM_CENTER, Y_CENTER),
        "right": (X_CENTER + X_FROM_CENTER, Y_CENTER),
    }
    # print(BUTTON_LAYOUT)

    buttons = []
    for action_name, (x, y) in BUTTON_LAYOUT.items():
        buttons.append(pygame.Rect(x, y, button_width, button_height))

    small_font = pygame.font.Font(None, 18)  # Smaller font size for buttons
    normal_font = pygame.font.Font(None, 25)


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


def send_discord_file(file_path, title, description):
    webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL)

    gen_file_name = file_path.name
    with open(file_path, "rb") as f:
        webhook.add_file(file=f.read(), filename=gen_file_name)

    embed = DiscordEmbed(
        title=title,
        description=description,
        color="03b2f8",
    )
    embed.set_thumbnail(url="attachment://" + gen_file_name)
    webhook.add_embed(embed)
    webhook.execute()


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

            if visualize:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                # 3. Show environment (the visual)
                # from (4, 240, 256, 3) to (240, 256, 3)
                frame = rbg_state[0]
                frame = np.transpose(frame, (1, 0, 2))
                surface = pygame.surfarray.make_surface(frame)
                scaled_surface = pygame.transform.scale(
                    surface, (game_width, game_height)
                )

                # Center the game display in the window
                screen.fill((0, 0, 0))  # Clear the screen with a black background
                screen.blit(scaled_surface, (offset_x, offset_y))

                # 4. Run agent on the state
            action = mario.act(state)

            if visualize:
                action_list = COMPLEX_MOVEMENT[action]
                # print(action_list)

                for i, button in enumerate(buttons):
                    # Draw button border
                    pygame.draw.ellipse(screen, (0, 0, 0), button.inflate(0, 1))

                    # Draw button background
                    if BUTTON_LIST[i] in action_list:
                        pygame.draw.ellipse(
                            screen, (255, 255, 0), button.inflate(-4, -4)
                        )  # Highlight active button

                        # print(BUTTON_LIST[i])
                        # print(button)
                    else:
                        pygame.draw.ellipse(
                            screen, (100, 100, 100), button.inflate(-4, -4)
                        )  # Inactive button

                    text_surface = small_font.render(
                        BUTTON_LIST[i], True, (255, 255, 255)
                    )
                    text_rect = text_surface.get_rect(center=button.center)
                    screen.blit(text_surface, text_rect)

            # 5. Agent performs action
            next_state, reward, done, info = env.step(action)
            mario.update_max(info)
            current_max = mario.get_max(info["world"], info["stage"])

            if visualize:
                hud_text = f"Episode: {e}, "
                if mario.curr_step < 1e5:
                    hud_text += f"Exploring mode, "
                else:
                    hud_text += "Learning mode, "
                hud_text += f"Step: {mario.curr_step}, Reward: {reward}"
                # compare with max_x_pos
                ratio = round(info["x_pos"] / current_max.get("x_pos", 1) * 100, 2)

                details_hub_text_1 = f"Max length: {current_max['x_pos']}, Max Score: {current_max['score']}, Max Coins: {current_max['coins']}"
                details_hub_text_2 = f"Max Time Left: {current_max['time_left']}, Best World: {mario.max_world}, Best Stage: {mario.max_stage}"
                details_hub_text_3 = f"Compare with best: {ratio}%"
                details_hub_text_4 = (
                    "Episode remaining: " + str(episodes - e) + " episodes"
                )

                hud_surface = normal_font.render(hud_text, True, (255, 255, 255))
                details_surface_1 = normal_font.render(
                    details_hub_text_1, True, (255, 255, 255)
                )
                details_surface_2 = normal_font.render(
                    details_hub_text_2, True, (255, 255, 255)
                )
                details_surface_3 = normal_font.render(
                    details_hub_text_3, True, (255, 255, 255)
                )
                details_surface_4 = normal_font.render(
                    details_hub_text_4, True, (255, 255, 255)
                )

                screen.blit(hud_surface, (10, 10))
                screen.blit(details_surface_1, (10, 40))
                screen.blit(details_surface_2, (10, 70))
                screen.blit(details_surface_3, (10, 100))
                screen.blit(details_surface_4, (10, 160))

                # Visualize weights
                # weights = mario.visualize_weights()
                # weights = (weights.cpu().numpy() * 255).astype(np.uint8)
                # for i in range(weights.shape[0]):
                #     weight_surface = pygame.surfarray.make_surface(weights[i])
                #     weight_surface = pygame.transform.scale(weight_surface, (84, 84))
                #     # print((game_width + 10 + (i % 4) * 90, 10 + (i // 4) * 90))
                #     screen.blit(
                #         weight_surface, (game_width + 10 + (i % 4) * 90, 10 + (i // 4) * 90)
                #     )

                pygame.display.update()
                clock.tick(60)

            # Calculate FPS
            # current_time = pygame.time.get_ticks()
            # elapsed_time = current_time - last_frame_time
            # last_frame_time = current_time
            # if elapsed_time > 0:
            #     fps = 1000 / elapsed_time
            #     print(f"FPS: {fps:.2f}")

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
                "Episode Rewards",
                f"current: {logger.ep_rewards[-1]}",
            )
            send_discord_file(
                logger.ep_lengths_plot,
                "Episode Lengths",
                f"current: {logger.ep_lengths[-1]}",
            )
            send_discord_file(
                logger.ep_avg_losses_plot,
                "Episode Avg Losses",
                f"current: {logger.ep_avg_losses[-1]}",
            )
            send_discord_file(
                logger.ep_avg_qs_plot,
                "Episode Avg Qs",
                f"current: {logger.ep_avg_qs[-1]}",
            )

        pbar.update()


# after the training is done, display the notification on the pygame window
if visualize:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((0, 0, 0))  # Clear the screen with a black background
        text_surface = normal_font.render("Training is done!", True, (255, 255, 255))
        text_rect = text_surface.get_rect(
            center=(window_width // 2, window_height // 2)
        )
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        clock.tick(60)
