import itertools as it

import numpy as np

import torch
import torch.nn as nn

from vizdoom import Mode

from tqdm import trange

from doom_env import create_doom_env, image_preprocessing, DEVICE
from doom_run import run
from doom_agent import DQNAgent

# Q-learing values
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# Neural Network Batch Size
batch_size = 64

# Other parameters
frame_repeat = 12
episodes_to_watch = 10

model_savefile = "./model/doom-model.pth"
save_model = True
load_model = False
skip_learning = False



if __name__ == '__main__':
    game = create_doom_env()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    agent = DQNAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                     memory_size=replay_memory_size, discount_factor=discount_factor,
                     load_model=load_model)

    if not skip_learning:
        agent, game = run(game, agent, actions, num_epochs=train_epochs, frame_repeat=frame_repeat,
                          steps_per_epoch=learning_steps_per_epoch, save_model=save_model, model_savefile=model_savefile)

        print("======================================")
        print("Training finished")

    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = image_preprocessing(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)