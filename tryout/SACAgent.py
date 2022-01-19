from BasicAgent import BasicAgent
from Networks import ActorNet, CriticNet
from Noise import OU_Noise
import torch
import torch.optim as optim
import torch.nn.functional as functional
from collections import deque
import numpy as np
import gym
import matplotlib.pyplot as plt
from utilities.data_structures.Config import Config
from torch.distributions import Normal

config = Config()
config.seed = 1
config.environment = gym.make("MountainCarContinuous-v0")
config.num_episodes_to_run = 450
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {
    "Actor": {
        "learning_rate": 0.003,
        "linear_hidden_units": [20, 20],
        "final_layer_activation": None,
        "batch_norm": False,
        "tau": 0.005,
        "gradient_clipping_norm": 5,
        "initialiser": "Xavier"
    },

    "Critic": {
        "learning_rate": 0.02,
        "linear_hidden_units": [20, 20],
        "final_layer_activation": None,
        "batch_norm": False,
        "buffer_size": 1000000,
        "tau": 0.005,
        "gradient_clipping_norm": 5,
        "initialiser": "Xavier"
    },
        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": True,
        "clip_rewards": False
}


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6
class SACAgent(BasicAgent):
    def __init__(self,
                 config,
                 load_model=False,
                 critic_checkpoint="critic.chkp",
                 actor_checkpoint="actor.chkp",
                 use_GPU=True,
                 actor_learning_rate=0.003,
                 critic_learning_rate=0.02,
                 gamma=.99,
                 eval_episodes=20,
                 episode_num=5000,
                 target_update_freq=1000,
                 replay_buffer_size=200000,
                 min_replay_size=None,
                 learning_updates_per_learning_session=10,
                 batch_size=64):
        super(SACAgent, self).__init__(config, use_GPU, gamma,
                                        eval_episodes, episode_num, replay_buffer_size, min_replay_size, batch_size)
        self.critic_checkpoint = critic_checkpoint
        self.actor_checkpoint = actor_checkpoint
        self.hyperparameters = config.hyperparameters
        self.action_types = "DISCRETE" if self.env.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = int(self.get_action_size())
        self.state_size = int(self.get_state_size())
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.target_update_freq = target_update_freq
        self.learning_updates_per_learning_session = learning_updates_per_learning_session
        self.actor_net = self.create_NN(
            self.state_size, self.action_size*2, key_to_use="Actor")
        self.critic_net = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_net_2 = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, override_seed=self.config.seed +1 , key_to_use="Critic")
        self.critic_target_net = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_target_net_2 = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=critic_learning_rate)
        self.critic_optimizer_2 = optim.Adam(
            self.critic_net_2.parameters(), lr=critic_learning_rate)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device = self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = optim.Adam([self.log_alpha], lr = self.hyperparameters["Actor"]["learning_rate"], eps = 1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        
        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
            self.hyperparameters["theta"], self.hyperparameters["sigma"])
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.step = 0
        self.best_score = None
        if load_model:
            self.load_checkpoint()
        self.copy_of_target_network(self.critic_net, self.critic_target_net)
        self.copy_of_target_network(self.critic_net_2, self.critic_target_net_2)

    def reset_game(self):
        BasicAgent.reset_game(self)
        if self.add_extra_noise: self.noise.reset()

    def get_state_size(self):
        """Gets the state_size for the gym env into the correct shape for a neural network"""
        random_state = self.env.reset()
        if isinstance(random_state, dict):
            state_size = random_state["observation"].shape[0] + \
                random_state["desired_goal"].shape[0]
            return state_size
        else:
            return random_state.size

    def get_action_size(self):
        """Gets the action_size for the gym env into the correct shape for a neural network"""
        if "action_size" in self.env.__dict__:
            return self.env.action_size
        if self.action_types == "DISCRETE":
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def compute_actor_loss(self, states):
        actions, log_pis, _ = self.get_action_info(states)
        qf1_pi = self.critic_net(torch.cat((states, actions), 1))
        qf2_pi = self.critic_net_2(torch.cat((states, actions), 1))
        mean_qf_pi = 0.5*(qf1_pi + qf2_pi)
        policy_loss = ((self.alpha * log_pis) - mean_qf_pi).mean()

        return policy_loss, log_pis

    def compute_critic_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_log_pis, _ = self.get_action_info(next_states)
            qf1_next_target = self.critic_target_net(torch.cat((next_states, next_actions), 1))
            qf2_next_target = self.critic_target_net_2(torch.cat((next_states, next_actions), 1))
            mean_qf_next_target = 0.5*(qf1_next_target + qf2_next_target) - self.alpha*next_log_pis
            next_q_value = rewards + (1.-dones)* self.gamma * mean_qf_next_target
        qf1 = self.critic_net(torch.cat((states, actions), 1))
        qf2 = self.critic_net_2(torch.cat((states, actions), 1))
        qf1_loss = functional.mse_loss(qf1, next_q_value)
        qf2_loss = functional.mse_loss(qf2, next_q_value)

        return qf1_loss, qf2_loss

    def pick_action_by_policy(self, state, eval):
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)    
        if eval: _, _ , action = self.get_action_info(state)
        else:
            with torch.no_grad():
                action, _ , _ = self.get_action_info(state)
                if self.add_extra_noise: action += torch.Tensor(self.noise.sample())

        action = action.detach().cpu().numpy()
        
        return action.squeeze(0)

    def get_action_info(self, state):
        actor_output = self.actor_net(state)
        mean, log_std = actor_output[:, : self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)     
        xt = normal.rsample()
        action = torch.tanh(xt)
        log_prob = normal.log_prob(xt)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def take_optimization_step(self, optimizer, local_net, loss, clipping_norm=None):
        optimizer.zero_grad()
        loss.backward()
        if clipping_norm is not None:
            # clip gradients to help stabilise training
            torch.nn.utils.clip_grad_norm_(
                local_net.parameters(), clipping_norm)
        optimizer.step()

    def update_actor(self, states):
        actor_loss, log_pis = self.compute_actor_loss(states)
        if self.automatic_entropy_tuning:
            self.update_entropy_parameters(log_pis)

        self.take_optimization_step(
            self.actor_optimizer, self.actor_net, actor_loss, self.hyperparameters["Actor"]["gradient_clipping_norm"])

    def update_entropy_parameters(self, log_pis):
        alpha_loss = self.compute_entropy_tuning_loss(log_pis)
        if alpha_loss is not None:
            self.take_optimization_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def update_critic(self, states, actions, rewards, next_states, dones):
        critic_loss_1, critic_loss_2 = self.compute_critic_loss(
            states, actions, rewards, next_states, dones)
        self.take_optimization_step(
            self.critic_optimizer, self.critic_net, critic_loss_1, self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimization_step(
            self.critic_optimizer_2, self.critic_net_2, critic_loss_2, self.hyperparameters["Critic"]["gradient_clipping_norm"])

        self.soft_update_of_target_network(self.critic_net, self.critic_target_net, self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_net_2, self.critic_target_net_2, self.hyperparameters["Critic"]["tau"])
    
    def compute_entropy_tuning_loss(self, log_pis):
        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
        return alpha_loss

    def time_to_learn(self):
        return self.step > self.hyperparameters["min_steps_before_learning"] and \
               len(self.replay_buffer) > self.batch_size and self.step % self.target_update_freq == 0

    def train(self):
        super(SACAgent, self).train()
        self.train_rewards = []
        self.eval_rewards = []
        for self.episode in range(self.episode_num):
            print("Episode {}".format(self.episode))
            self.reset_game()
            eval = self.episode % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
            score = self.run_one_episode(eval)
            if eval: 
                self.eval_rewards.append(score)
                print("Evaluated score: ", score)
            else : 
                self.train_rewards.append(score)
                average_reward = np.mean(self.train_rewards[-self.eval_episodes:])
                print("Reward: ", average_reward)
                if self.best_score is None or self.best_score < average_reward:
                    self.save_checkpoint()
                    print("Best average reward: ", average_reward)
                    self.best_score = average_reward
        
    def run_one_episode(self, eval):
        score = 0
        while not self.done:
            self.action = self.pick_action_by_policy(self.state, eval = eval)
            self.next_state, self.reward, self.done, _ = self.env.step(self.action)
            score += self.reward
            #if not eval and self.reward > 0: print("Good reward: ", self.reward)
            self.step += 1
            if self.time_to_learn():
                for _ in range(self.learning_updates_per_learning_session):
                    self.perform_one_step_training()
            mask = False if self.episode >= self.env._max_episode_steps else self.done
            if not eval: self.replay_buffer.push(self.state, self.action, self.next_state, self.reward, mask)
            self.state = self.next_state

        return score

    def perform_one_step_training(self):
        # pick a batch randomly from the replay buffer
        states, actions, rewards, next_states, dones = self.sample_experiences()
        # update critic parameters
        self.update_critic(states, actions, rewards, next_states, dones)
        # update actor parameters
        self.update_actor(states)

    def sample_experiences(self, size = None):
        if size is None: size = self.batch_size
        experiences = self.replay_buffer.sample(size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.critic_net.state_dict(), self.critic_checkpoint)
        torch.save(self.actor_net.state_dict(), self.actor_checkpoint)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.critic_net.load_state_dict(
            torch.load(self.critic_checkpoint), strict=False)
        self.actor_net.load_state_dict(
            torch.load(self.actor_checkpoint), strict=False)

    def evaluate(self, episodes = None):
        if episodes is None: episodes = self.eval_episodes
        episode_rewards = deque([0.], maxlen=episodes)
        with torch.no_grad():
            for _ in range(episodes):
                episode_reward = 0.
                state = self.env.reset()
                done = False
                while not done:
                    action = self.pick_action_by_policy(state, eval = True)
                    next_state, reward, done, _ = self.env.step(action)
                    state = next_state
                    episode_reward += reward
     
                episode_rewards.append(episode_reward)
        return np.mean(episode_rewards), np.max(episode_rewards)

    def visualize_episode(self, rewards, title):
        _, ax = plt.subplots()
        ax.set_ylabel(title)
        ax.set_xlabel('Number of episodes')
        ax.plot(rewards, label='Average rewards')
        plt.show()


if __name__ == "__main__":
    #config.hyperparameters["Actor"]["y_range"] = (np.float32(env.action_space.low[0]-0.25).item(), np.float32(env.action_space.high[0]+0.25).item())
    
    agent = SACAgent(config, load_model=True, use_GPU=False,
                      replay_buffer_size=1000000, batch_size=256, target_update_freq=50, eval_episodes = 50, episode_num=5000)
    agent.train()
    #agent.load_checkpoint()
    agent.visualize_episode(agent.train_rewards, "Training rewards")
    agent.visualize_episode(agent.eval_rewards, "Evaluated rewards")
