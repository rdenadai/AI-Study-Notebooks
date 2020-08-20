import os, sys
import time
import pickle
import numpy as np
from PIL import Image
import pygame
from code.lib.utils import randargmax


N_ACTIONS = 4

PLAYER_POS = {"x": 0, "y": 0}

PLAYER = 1
FOOD = 2
WALL = 3
ENEMY = 4


WIN_SIZE = (150, 150)


def move(x, y, size, action, world):
    ox, oy = x, y
    if action == 0:
        if y > 0:
            y -= 1
    elif action == 1:
        if y < size - 1:
            y += 1
    elif action == 2:
        if x > 0:
            x -= 1
    elif action == 3:
        if x < size - 1:
            x += 1
    if world[y, x] == WALL:
        return ox, oy
    return x, y


def get_pygame_surface(world):
    world[world == PLAYER] = 255
    world[world == FOOD] = 160
    world[world == WALL] = 50
    world[world == ENEMY] = 100
    world = np.dstack([world for _ in range(3)]).astype(np.uint8)
    image = Image.fromarray(world, "RGB")
    image = image.resize(WIN_SIZE)
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)


if __name__ == "__main__":
    try:
        with open("code/qlearning/files/world.pkl", "rb") as f:
            zero_world = pickle.load(f)
        size = len(zero_world[0, :])
        # Q-Table
        with open("code/qlearning/files/qtable.pkl", "rb") as f:
            Q = pickle.load(f)

        pygame.init()
        clock = pygame.time.Clock()
        display = pygame.display.set_mode(WIN_SIZE, pygame.DOUBLEBUF)
        pygame.display.set_caption("Q-Learning")

        state = 0
        reward = 0
        stop = False
        world = np.copy(zero_world)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            display.fill((0, 0, 0))
            display.blit(get_pygame_surface(np.copy(world)), (0, 0))
            pygame.display.update()
            clock.tick(10)

            win = False

            # Player actual position
            x, y = PLAYER_POS.get("x", 0), PLAYER_POS.get("y", 0)

            # Action
            action = randargmax(Q[state, :])
            x, y = move(x, y, size, action, world)

            # Collision
            if world[y, x] == FOOD:
                reward += 1
            elif world[y, x] == ENEMY:
                stop = True

            # No food... stop
            if not np.any(world == FOOD):
                stop = True
                win = True

            # Update player position
            world[PLAYER_POS.get("y", 0), PLAYER_POS.get("x", 0)] = 0
            PLAYER_POS["x"] = x
            PLAYER_POS["y"] = y
            world[y, x] = PLAYER

            # New state
            state = x + (y * size)

            if stop:
                print(f"Game: {stop}")
                print("-" * 20)
                if win:
                    print(f"**** WIN ****")
                print(f"Reward : {np.round(reward, 2)}")
                print("-" * 20)

                time.sleep(1)
                os.system("clear")

                PLAYER_POS = {"x": 0, "y": 0}
                state, reward = 0, 0
                stop = False
                world = np.copy(zero_world)
    except KeyboardInterrupt:
        pass
