import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import random, datetime
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

# Initialize Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

# Limit the action-space to
#   0. walk right
#   1. jump right
# env = JoypadSpace(env, [["right"], ["right", "A"], ["left"], ["left", "A"]])
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# Apply Wrappers to environment
env = SkipFrame(env, skip=1)

# rbg_display is used to display the game
rbg_display = env

env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.0)
env = FrameStack(env, num_stack=1)
env.reset()

# Create a separate environment for display
rbg_display = FrameStack(rbg_display, num_stack=4)
rbg_display.reset()

pygame.init()


screen = pygame.display.set_mode((256 * 2, 240 * 2 * 2))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)

BUTTON_LIST = ["A", "B", "right", "left", "down", "up"]

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

checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
mario = Mario(
    state_dim=(1, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint,
)

logger = MetricLogger(save_dir)

episodes = 40000
worlds = range(1, 9)
stages = range(1, 5)

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()
    rbg_state = rbg_display.reset()

    last_x_pos_screen = 0
    last_x_pos_screen_time = pygame.time.get_ticks()

    # Play the game!
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # 3. Show environment (the visual)
        # from (4, 240, 256, 3) to (240, 256, 3)
        frame = rbg_state[0]
        frame = np.transpose(frame, (1, 0, 2))
        surface = pygame.surfarray.make_surface(frame)
        scaled_surface = pygame.transform.scale(surface, (game_width, game_height))

        # Center the game display in the window
        screen.fill((0, 0, 0))  # Clear the screen with a black background
        screen.blit(scaled_surface, (offset_x, offset_y))

        # 4. Run agent on the state
        action = mario.act(state)
        action_list = COMPLEX_MOVEMENT[action]

        for i, button in enumerate(buttons):
            # Draw button border
            pygame.draw.ellipse(screen, (0, 0, 0), button.inflate(0, 1))

            # Draw button background
            if BUTTON_LIST[i] in action_list:
                pygame.draw.ellipse(
                    screen, (255, 255, 0), button.inflate(-4, -4)
                )  # Highlight active button
            else:
                pygame.draw.ellipse(
                    screen, (100, 100, 100), button.inflate(-4, -4)
                )  # Inactive button

            text_surface = small_font.render(BUTTON_LIST[i], True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=button.center)
            screen.blit(text_surface, text_rect)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)
        mario.update_max(info)
        current_max = mario.get_max(info["world"], info["stage"])

        hud_text = f"Episode: {e}, "
        if mario.curr_step < 1e5:
            hud_text += f"Exploring mode, "
        else:
            hud_text += "Learning mode, "
        hud_text += f"Step: {mario.curr_step}, Reward: {reward}"

        details_hub_text_1 = (
            f"Max X pos: {current_max['x_pos']}, Max Score: {current_max['score']}"
        )
        details_hub_text_2 = f"Max Coins: {current_max['coins']}, Max Time Left: {current_max['time_left']}"

        hud_surface = normal_font.render(hud_text, True, (255, 255, 255))
        details_surface_1 = normal_font.render(
            details_hub_text_1, True, (255, 255, 255)
        )
        details_surface_2 = normal_font.render(
            details_hub_text_2, True, (255, 255, 255)
        )
        screen.blit(hud_surface, (10, 10))
        screen.blit(details_surface_1, (10, 40))
        screen.blit(details_surface_2, (10, 70))

        pygame.display.flip()
        clock.tick(60)

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

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

# after the training is done, display the notification on the pygame window
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.fill((0, 0, 0))  # Clear the screen with a black background
    text_surface = normal_font.render("Training is done!", True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    clock.tick(60)
