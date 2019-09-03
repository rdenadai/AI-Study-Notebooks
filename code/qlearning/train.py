import os
import time
import pickle
import numpy as np
from code.lib.utils import randargmax, normalize


EPISODES = 5_000
MAX_STEPS = 99

N_ACTIONS = 4
LEARNING_RATE = 0.1
GAMMA = 0.3

DECAY_RATE = 0.001
EPSILON = 0.99
MAX_EPSILON = EPSILON
MIN_EPSILON = 0.1


PLAYER_POS = {"x": 0, "y": 0}

PLAYER = 1
FOOD = 2
ENEMY = 3


def generate_random_action(x, y, size):
    return np.random.randint(N_ACTIONS)


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


def generate_world():
    F, E = FOOD, ENEMY
    # return np.array(
    #     [
    #         [0, 0, 0, 0, F],
    #         [E, E, E, 0, 0],
    #         [0, 0, F, 0, 0],
    #         [E, 0, E, 0, E],
    #         [0, F, 0, 0, 0],
    #     ]
    # )
    return np.array([[0, 0, E], [E, 0, F], [F, 0, 0]])


if __name__ == "__main__":
    try:
        zero_world = generate_world()
        size = np.max(zero_world.shape)
        # Q-Table
        try:
            with open("code/qlearning/files/qtable.pkl", "rb") as f:
                Q = pickle.load(f)
        except Exception as e:
            extra_state = len(zero_world[zero_world == FOOD])
            Q = np.zeros((size ** 2 + extra_state, N_ACTIONS))
            # Q = -np.ones((size ** 2 + extra_state, N_ACTIONS))

        for episode in range(EPISODES):
            PLAYER_POS = {"x": 0, "y": 0}
            state, action, t_reward, reward = 0, 0, 0, 0
            win = False
            stop = False
            world = np.copy(zero_world)

            result = np.where(world == FOOD)
            coords = list(zip(result[0], result[1]))
            food_position = {}
            for i, position in enumerate(coords):
                food_position.update({position: (size ** 2) + i})

            for k in range(MAX_STEPS):
                # Player actual position
                x, y = PLAYER_POS.get("x", 0), PLAYER_POS.get("y", 0)

                # Action
                action = generate_random_action(x, y, size)
                if np.random.uniform(0, 1) > EPSILON:
                    action = randargmax(Q[state, :])
                x, y = move(x, y, size, action)

                # New state
                new_state = x + (y * size)

                # Clear reward
                reward = 0
                # Collision
                if world[y, x] == FOOD:
                    reward = 25
                    t_reward += 1
                    new_state = food_position[(y, x)]
                elif world[y, x] == ENEMY:
                    reward = -100
                    stop = True
                else:
                    reward = -1

                # No food... stop
                if not np.any(world == FOOD):
                    stop = True
                    win = True

                # Update player position
                world[PLAYER_POS.get("y", 0), PLAYER_POS.get("x", 0)] = 0
                PLAYER_POS["x"] = x
                PLAYER_POS["y"] = y
                world[y, x] = PLAYER

                # Update Q-Table
                target = LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]))
                if win:
                    Q[state, action] += target
                else:
                    Q[state, action] = (1 - LEARNING_RATE) * Q[state, action] + target

                # Pass new state
                state = new_state

                if stop:
                    break

            print(f"Episode: {episode}")
            print(f"Reward : {np.round(t_reward, 2)}")
            print(f"Epsilon: {EPSILON}")
            print("-" * 20)
            time.sleep(0.01)
            os.system("clear")

            EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(
                -DECAY_RATE * episode
            )
    except KeyboardInterrupt:
        pass

    print(Q)
    with open("code/qlearning/files/world.pkl", "wb") as f:
        pickle.dump(np.copy(zero_world), f)
    with open("code/qlearning/files/qtable.pkl", "wb") as f:
        pickle.dump(Q, f)
