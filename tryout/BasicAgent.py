from Replay_Buffer import Replay_Buffer
import os
import torch
import numpy as np
import random
import gym
from nn_builder.pytorch.NN import NN
from collections import deque


class BasicAgent(object):
    def __init__(self, config, load_model, use_GPU, gamma, eval_episodes, eval_episode_freq, episode_num, replay_buffer_size, min_replay_size, batch_size) -> None:
        self.env = config.environment
        self.config = config
        self.eval_episodes = eval_episodes
        self.gamma = gamma
        self.episode_num = episode_num
        self.load_model = load_model
        self.eval_episode_freq = eval_episode_freq
        self.device = "cuda:0" if use_GPU else "cpu"
        self.min_replay_size = min_replay_size if min_replay_size is not None else batch_size
        self.replay_buffer = Replay_Buffer(
            config.seed, replay_buffer_size, self.device)
        self.batch_size = batch_size
        self.set_random_seeds(config.seed)
        self.score_required_to_win = self.get_score_required_to_win()
        self.do_evaluation_iterations = self.config.hyperparameters["do_evaluation_iterations"]
        self.step = 0
        self.best_score = None

        # stops it from printing an unnecessary warning
        gym.logger.set_level(40)

    def save_checkpoint(self):
        raise NotImplementedError()

    def load_checkpoint(self):
        raise NotImplementedError()

    def initialize_replay_buffer(self):
        for _ in range(self.min_replay_size):
            state = self.env.reset()
            done = False
            while not done:
                 action = self.pick_random_action(state)
                 next_state, reward, done, _ = self.env.step(action)
                 self.replay_buffer.push(
                     state, action, next_state, reward, done)
                 state = next_state

    def pick_random_action(self, state=None):
        raise NotImplementedError()

    def pick_action_by_policy(self, state, eval):
        raise NotImplementedError()

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def copy_of_target_network(self, local_model, target_model):
        """Copy the target network in the direction of the local network"""
        target_model.load_state_dict(local_model.state_dict())

    def get_score_required_to_win(self):
        try: return self.env.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.env.spec.reward_threshold
            except AttributeError:
                return self.env.unwrapped.spec.reward_threshold

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """Creates a neural network for the agents to use"""
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        if key_to_use:
            hyperparameters = hyperparameters[key_to_use]
        if override_seed:
            seed = override_seed
        else:
            seed = self.config.seed

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.env.seed(self.config.seed)
        self.state = self.env.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        if "noise" in self.__dict__.keys(): self.noise.reset()

    def train(self):
        if self.load_model: self.load_checkpoint()
        self.train_rewards = []
        self.eval_rewards = []
        for self.episode in range(self.episode_num):
            self.train_one_episode()

    def train_one_episode(self):
        print("Episode {}".format(self.episode))
        self.reset_game()
        eval = self.episode % self.eval_episode_freq == 0 and self.do_evaluation_iterations
        score = self.run_one_episode(eval)
        if eval:
            self.eval_rewards.append(score)
            print("Evaluated score: ", score)
        else:
            self.train_rewards.append(score)
            average_reward = np.mean(
                self.train_rewards[-self.eval_episodes:])
            print("Reward: ", average_reward)
            if self.best_score is None or self.best_score < average_reward:
                self.save_checkpoint()
                print("Best average reward: ", average_reward)
                self.best_score = average_reward
       
    def run_one_episode(self, eval):
        score = 0
        while not self.done:
            self.action = self.pick_action_by_policy(self.state, eval=eval)
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            score += self.reward
            #if not eval and self.reward > 0: print("Good reward: ", self.reward)
            self.step += 1
            if self.time_to_learn():
                for _ in range(self.learning_updates_per_learning_session):
                    self.perform_one_step_training()
            mask = False if self.episode >= self.env._max_episode_steps else self.done
            if not eval:
                self.replay_buffer.push(
                    self.state, self.action, self.next_state, self.reward, mask)
            self.state = self.next_state

        return score

    def evaluate(self, episodes=None):
        if episodes is None:
            episodes = self.eval_episodes
        episode_rewards = deque([0.], maxlen=episodes)
        with torch.no_grad():
            for _ in range(episodes):
                episode_reward = 0.
                state = self.env.reset()
                done = False
                while not done:
                    action = self.pick_action_by_policy(state, eval=True)
                    next_state, reward, done, _ = self.env.step(action)
                    state = next_state
                    episode_reward += reward

                episode_rewards.append(episode_reward)
        return np.mean(episode_rewards), np.max(episode_rewards)
