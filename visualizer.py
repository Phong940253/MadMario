import pygame
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


def initialize_pygame(window_width, window_height):
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
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

    buttons = []
    for action_name, (x, y) in BUTTON_LAYOUT.items():
        buttons.append(pygame.Rect(x, y, button_width, button_height))

    small_font = pygame.font.Font(None, 18)  # Smaller font size for buttons
    normal_font = pygame.font.Font(None, 25)
    return (
        screen,
        clock,
        font,
        BUTTON_LIST,
        buttons,
        small_font,
        normal_font,
        BUTTON_LAYOUT,
    )


def visualize_game(
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
):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    # 3. Show environment (the visual)
    # from (4, 240, 256, 3) to (240, 256, 3)
    for frame in rbg_state:
        frame = rbg_state[0]
        frame = np.transpose(frame, (1, 0, 2))
        surface = pygame.surfarray.make_surface(frame)
        scaled_surface = pygame.transform.scale(surface, (game_width, game_height))

        # Center the game display in the window
        screen.fill((0, 0, 0))  # Clear the screen with a black background
        screen.blit(scaled_surface, (offset_x, offset_y))

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

        hud_text = f"Episode: {e}, "
        if mario.curr_step < 1e5:
            hud_text += f"Exploring mode, "
        else:
            hud_text += "Learning mode, "
        hud_text += f"Step: {mario.curr_step}, Reward: {info.get('reward', 0)}"
        # compare with max_x_pos
        ratio = round(info["x_pos"] / current_max.get("x_pos", 1) * 100, 2)

        details_hub_text_1 = f"Max length: {current_max['x_pos']}, Max Score: {current_max['score']}, Max Coins: {current_max['coins']}"
        details_hub_text_2 = f"Max Time Left: {current_max['time_left']}, Best World: {mario.max_world}, Best Stage: {mario.max_stage}"
        details_hub_text_3 = f"Compare with best: {ratio}%"
        details_hub_text_4 = "Episode remaining: " + str(100000 - e) + " episodes"

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

        pygame.display.update()
        clock.tick(60)


def display_training_done(screen, clock, normal_font, window_width, window_height):
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


def display_switching_checkpoint(
    screen, clock, normal_font, window_width, window_height
):
    screen.fill((0, 0, 0))  # Clear the screen with a black background
    text_surface = normal_font.render(
        "Syncing with the latest checkpoint...", True, (255, 255, 255)
    )
    text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    clock.tick(60)
