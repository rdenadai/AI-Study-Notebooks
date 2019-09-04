import os
import time
import pickle
import numpy as np
from code.lib.utils import randargmax


N_ACTIONS = 4

PLAYER_POS = {"x": 0, "y": 0}

PLAYER = 1
FOOD = 2
ENEMY = 3


def move(x, y, size, action):
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
    return x, y


if __name__ == "__main__":
    try:
        with open("code/qlearning/files/world.pkl", "rb") as f:
            zero_world = pickle.load(f)
        size = len(zero_world[0, :])
        # Q-Table
        with open("code/qlearning/files/qtable.pkl", "rb") as f:
            Q = pickle.load(f)

        state = 0
        reward = 0
        stop = False
        world = np.copy(zero_world)

        while True:
            for _ in range(50):
                print(world)
                time.sleep(0.25)
                os.system("clear")
                win = False

                # Player actual position
                x, y = PLAYER_POS.get("x", 0), PLAYER_POS.get("y", 0)

                # Action
                action = randargmax(Q[state, :])
                x, y = move(x, y, size, action)

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
                    break
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
