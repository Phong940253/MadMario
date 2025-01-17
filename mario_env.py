import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Create the environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, [["NOOP"], ["up"], ["down"], ["left"], ["right"], ["A"], ["B"]])

# Reset the environment
state = env.reset()

# Get the screen dimensions from the environment
height, width, _ = env.observation_space.shape

# Create the game window
screen = pygame.display.set_mode((width * 2, height * 2))
pygame.display.set_caption("Mario Environment")

# Game loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle keyboard input
    action = 0
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            action = 3  # Left
        elif event.key == pygame.K_RIGHT:
            action = 4  # Right
        elif event.key == pygame.K_UP:
            action = 5  # Jump
        elif event.key == pygame.K_DOWN:
            action = 6  # Down
        elif event.key == pygame.K_a:
            action = 5  # A button
        elif event.key == pygame.K_s:
            action = 6  # B button
        elif event.key == pygame.K_SPACE:
            action = 1  # Jump Right
        else:
            action = 0  # No action

    state, reward, done, info = env.step(action)

    # Convert the state to a Pygame surface
    state = np.transpose(state, (1, 0, 2))
    surface = pygame.surfarray.make_surface(state)

    # Scale the surface
    scaled_surface = pygame.transform.scale(surface, (width * 2, height * 2))

    # Draw the surface to the screen
    screen.blit(scaled_surface, (0, 0))

    # Update the display
    pygame.display.flip()
    clock.tick(60)

    if done:
        state = env.reset()

# Quit Pygame
pygame.quit()
env.close()
