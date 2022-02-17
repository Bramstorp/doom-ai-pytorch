import torch

from tqdm import trange
from time import time
import numpy as np

from doom_env import image_preprocessing

frame_repeat = 12

# Training regime
test_episodes_per_epoch = 100

def test(game, agent, actions):
    print("\nAi Testing...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = image_preprocessing(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        reward = game.get_total_reward()
        test_scores.append(reward)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())

def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=0, save_model=False, model_savefile=""):
    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch, leave=False):
            state = image_preprocessing(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = image_preprocessing(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        test(game, agent, actions)
        if save_model:
            print("Saving the network to model file:", model_savefile)
            torch.save(agent.q_net, model_savefile)
        print(f"Total elapsed time: {((time() - start_time) / 60.0)} minutes")

    game.close()
    return agent, game