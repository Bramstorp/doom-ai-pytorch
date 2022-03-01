import os

import itertools as it
from vizdoom import Mode
from time import sleep

from doom_env import create_doom_env, image_preprocessing
from helper import plot

# DQN
from models.dqn.doom_train import train
from models.dqn.doom_agent import DQNAgent


# Q-learing values
class Parameters():
    def __init__(self, learning_rate=0.00025, discount_factor=0.99, train_epochs=20, learning_steps_per_epoch=2000, replay_memory_size=10000, batch_size=64, frame_repeat=12, config_file_path= "scenarios/deadly_corridor.cfg", learing=False, testing=True, episodes=10):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.train_epochs = train_epochs
        self.learning_steps_per_epoch = learning_steps_per_epoch
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.frame_repeat = frame_repeat

        self.model_savefile = f"./trained_models/{train_epochs}/doom-model.pth"
        self.config_file_path = config_file_path

        if learing:
            self.save_model = True
            self.load_model = False
            self.skip_learning = False

        if testing:
            self.save_model = False
            self.load_model = True
            self.skip_learning = True

        if not testing and not learing:
            self.save_model = False
            self.load_model = False
            self.skip_learning = False

        # Testing Episodes to watch
        self.episodes = episodes


def run_ai():
    values = Parameters()

    plot_scores = []
    total_score = 0
    game = create_doom_env(values.config_file_path)
    numbers_of_available_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=numbers_of_available_buttons)]

    path = f"trained_models/{values.train_epochs}"
    isExist = os.path.exists(path)

    agent = DQNAgent(len(actions), lr=values.learning_rate, batch_size=values.batch_size,
                     memory_size=values.replay_memory_size, discount_factor=values.discount_factor,
                     load_model=values.load_model, model_savefile=values.model_savefile)

    if not values.skip_learning and not isExist:
        os.makedirs(path)
        if path:
            agent, game = train(game, agent, actions, num_epochs=values.train_epochs, frame_repeat=values.frame_repeat,steps_per_epoch=values.learning_steps_per_epoch, save_model=values.save_model, model_savefile=values.model_savefile)

            print("****************************")
            print("Training finished")
            print("****************************")

    if isExist and values.skip_learning:
        game.close()
        game.set_window_visible(True)
        game.set_mode(Mode.ASYNC_PLAYER)
        game.init()

        for values.episode in range(values.episodes):
            game.new_episode()
            while not game.is_episode_finished():
                state = image_preprocessing(game.get_state().screen_buffer)
                best_action_index = agent.get_action(state)

                game.set_action(actions[best_action_index])
                for _ in range(values.frame_repeat):
                    game.advance_action()

            sleep(1.0)
            score = game.get_total_reward()
            print("Total score: ", score)

            # Plot scored with amount of game + mean
            plot_scores.append(score)
            plot(plot_scores)

    if isExist and not values.skip_learning:
        print("The model already exist at that epoch so u cannot train")