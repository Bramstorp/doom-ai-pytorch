import itertools as it
import numpy as np
import torch.nn as nn
from vizdoom import Mode
from time import sleep

from doom_env import create_doom_env, image_preprocessing
from helper import plot

# DQN
from models.dqn.doom_train import train
from models.dqn.doom_agent import DQNAgent


# Q-learing values
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 50
learning_steps_per_epoch = 2000
replay_memory_size = 10000
batch_size = 64
frame_repeat = 12

model_savefile = "./trained_models/50/doom-model.pth"
config_file_path = "scenarios/deadly_corridor.cfg"
LOG_DIR = './logs/log_corridor'
save_model = True
load_model = False
skip_learning = False
show_while_learning = False
testing_after_run = False

# Testing Episodes to watch
episodes = 50

if __name__ == '__main__':
    plot_scores = []
    total_score = 0

    game = create_doom_env(config_file_path)
    numbers_of_available_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=numbers_of_available_buttons)]

    agent = DQNAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                     memory_size=replay_memory_size, discount_factor=discount_factor,
                     load_model=load_model, model_savefile=model_savefile)

    if not skip_learning:
        agent, game = train(game, agent, actions, num_epochs=train_epochs, frame_repeat=frame_repeat,
                          steps_per_epoch=learning_steps_per_epoch, save_model=save_model, model_savefile=model_savefile, testing=testing_after_run)

        print("****************************")
        print("Training finished")
        print("****************************")

    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for episode in range(episodes):
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

        # Plot scored with amount of game + mean
        plot_scores.append(score)
        plot(plot_scores)