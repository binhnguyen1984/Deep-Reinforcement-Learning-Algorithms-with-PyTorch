from BasicAgent import BasicAgent
from Networks import ActorNet, CriticNet
from Noise import OU_Noise
import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
import gym
import matplotlib.pyplot as plt
from utilities.data_structures.Config import Config


config = Config()
config.seed = 1
config.environment = gym.make("BipedalWalker-v3")
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
        "linear_hidden_units": [400, 300],
        "final_layer_activation": "tanh",
        "batch_norm": False,
        "tau": 0.005,
        "gradient_clipping_norm": 5,
        "initialiser": "Xavier"
    },

    "Critic": {
        "learning_rate": 0.02,
        "linear_hidden_units": [400, 300],
        "final_layer_activation": None,
        "batch_norm": False,
        "buffer_size": 1000000,
        "tau": 0.005,
        "gradient_clipping_norm": 5,
        "initialiser": "Xavier"
    },
    "batch_size": 256,
    "discount_rate": 0.99,
    "mu": 0.0,  # for O-H noise
    "theta": 0.15,  # for O-H noise
    "sigma": 0.25,  # for O-H noise
    "action_noise_std": 0.2,  # for TD3
    "action_noise_clipping_range": 0.5,  # for TD3
    "update_every_n_steps": 50,
    "learning_updates_per_learning_session": 10,
    "automatically_tune_entropy_hyperparameter": True,
    "entropy_term_weight": None,
    "add_extra_noise": True,
    "do_evaluation_iterations": True,
    "clip_rewards": False    
}

TRAINING_EPISODES_PER_EVAL_EPISODE = 10
class DDPQAgent(BasicAgent):
    def __init__(self,
                 config,
                 load_model=False,
                 critic_checkpoint="critic.chkp",
                 actor_checkpoint="actor.chkp",
                 use_GPU=True,
                 actor_learning_rate=0.003,
                 critic_learning_rate=0.02,
                 gamma=.99,
                 tau=0.005,
                 eval_episodes=20,
                 eval_episode_freq = 10,
                 episode_num=5000,
                 target_update_freq=1000,
                 replay_buffer_size=200000,
                 min_replay_size=None,
                 learning_updates_per_learning_session=10,
                 batch_size=64):
        super(DDPQAgent, self).__init__(config, load_model, use_GPU, gamma,
                                        eval_episodes, eval_episode_freq, episode_num, replay_buffer_size, min_replay_size, batch_size)
        self.critic_checkpoint = critic_checkpoint
        self.actor_checkpoint = actor_checkpoint
        self.hyperparameters = config.hyperparameters
        self.action_types = "DISCRETE" if self.env.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = int(self.get_action_size())
        self.state_size = int(self.get_state_size())
        self.noise = OU_Noise(self.action_size, config.seed)
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.target_update_freq = target_update_freq
        self.learning_updates_per_learning_session = learning_updates_per_learning_session
        self.actor_net = self.create_NN(
            self.state_size, self.action_size, key_to_use="Actor")
        self.actor_target_net = self.create_NN(
            self.state_size, self.action_size, key_to_use="Actor")
        self.critic_net = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_target_net = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=critic_learning_rate)
        self.copy_of_target_network(self.actor_net, self.actor_target_net)
        self.copy_of_target_network(self.critic_net, self.critic_target_net)


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

    def compute_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor_net(state).cpu().data.numpy()
        return action
    
    def pick_random_action(self, state):
        action = self.compute_action(state)
        action += self.noise.sample()
        return action.squeeze(0)

    def pick_action_by_policy(self, state, eval):
        if eval: return self.compute_action(state).squeeze(0)
        else: return self.pick_random_action(state)
    
    def compute_actor_loss(self, states):
        actions_pred = self.actor_net(states)
        return -self.critic_net(torch.cat((states, actions_pred),1)).mean()

    def compute_critic_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(
                next_states, rewards, dones)

        critic_expected = self.compute_critic_expected_values(states, actions)
        return functional.mse_loss(critic_expected, critic_targets)

    def compute_critic_expected_values(self, states, actions):
        return self.critic_net(torch.cat((states, actions),1))

    def compute_critic_values_for_next_states(self, next_states):
        with torch.no_grad():          
            actions_next = self.actor_target_net(next_states)
            critic_targets_next = self.critic_target_net(torch.cat((
                next_states, actions_next),1))

        return critic_targets_next

    def compute_critic_targets(self, next_states, rewards, dones):
        critic_targets_next = self.compute_critic_values_for_next_states(
            next_states)
        return rewards + self.gamma*critic_targets_next*(1.0-dones)

    def take_learning_step(self, optimizer, local_net, target_net, loss, clipping_norm=None):
        optimizer.zero_grad()
        loss.backward()
        if clipping_norm is not None:
            # clip gradients to help stabilise training
            torch.nn.utils.clip_grad_norm_(
                local_net.parameters(), clipping_norm)
        optimizer.step()
        self.soft_update_of_target_network(local_net, target_net, self.tau)

    def update_actor(self, states):
        actor_loss = self.compute_actor_loss(states)
        self.take_learning_step(
            self.actor_optimizer, self.actor_net, self.actor_target_net, actor_loss, self.hyperparameters["Actor"]["gradient_clipping_norm"])

    def update_critic(self, states, actions, rewards, next_states, dones):
        critic_loss = self.compute_critic_loss(
            states, actions, rewards, next_states, dones)
        self.take_learning_step(
            self.critic_optimizer, self.critic_net, self.critic_target_net, critic_loss, self.hyperparameters["Critic"]["gradient_clipping_norm"])

    def time_to_learn(self):
        return len(self.replay_buffer) > self.batch_size and self.step % self.target_update_freq == 0

    def train_one_episode(self):
        super(DDPQAgent, self).train_one_episode()
        self.update_learning_rate(self.actor_learning_rate, self.actor_optimizer)
        self.update_learning_rate(self.critic_learning_rate, self.critic_optimizer)
        
    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.train_rewards) > 0:
            last_rolling_score = np.max(self.train_rewards)
            if last_rolling_score > 0.75 * self.score_required_to_win:
                starting_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.score_required_to_win:
                starting_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.score_required_to_win:
                starting_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.score_required_to_win:
                starting_lr = starting_lr / 2.0
            for g in optimizer.param_groups:
                g['lr'] = starting_lr

    def perform_one_step_training(self):
        # pick a batch randomly from the replay buffer
        states, actions, rewards, next_states, dones = self.sample_experiences()
        # update actor parameters
        self.update_actor(states)
        # update critic parameters
        self.update_critic(states, actions, rewards, next_states, dones)

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

    def visualize_episode(self, rewards):
        fig, ax = plt.subplots()
        ax.set_ylabel('Rewards over a number of episodes')
        ax.set_xlabel('Number of episodes')
        ax.plot(rewards, label='Average rewards')
        plt.show()


if __name__ == "__main__":
    #config.hyperparameters["Actor"]["y_range"] = (np.float32(env.action_space.low[0]-0.25).item(), np.float32(env.action_space.high[0]+0.25).item())
    
    agent = DDPQAgent(config, load_model=False, use_GPU=False,
                     replay_buffer_size=1000000, batch_size=256, target_update_freq=50, eval_episodes=50, episode_num=5000)
    agent.train()
    #agent.load_checkpoint()
    agent.visualize_episode(agent.train_rewards)
