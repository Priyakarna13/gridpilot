import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import namedtuple, deque

# Self-contained experiment runner (duplicates core classes to avoid import side effects).

class GridWorld:
    """3x4 stochastic GridWorld environment."""
    def __init__(self, trap_penalty=-1.0):
        self.height = 3
        self.width = 4
        self.start_state = (2, 0)
        self.goal_state = (0, 3)
        self.trap_state = (1, 3)
        self.block_state = (1, 1)
        self.terminal_states = [self.goal_state, self.trap_state]

        self.rewards = np.zeros((self.height, self.width))
        self.rewards[self.goal_state] = 1.0
        self.rewards[self.trap_state] = trap_penalty

        self.living_cost = -0.04

        self.actions = ['N', 'E', 'S', 'W']
        self.action_map = { 0: 'N', 1: 'E', 2: 'S', 3: 'W' }
        self.current_state = self.start_state

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def _get_transition_probs(self, action_char):
        idx = self.actions.index(action_char)
        left = self.actions[(idx - 1) % 4]
        right = self.actions[(idx + 1) % 4]
        return {action_char: 0.8, left: 0.1, right: 0.1}

    def _move(self, state, action_char):
        i, j = state
        if action_char == 'N': i -= 1
        elif action_char == 'E': j += 1
        elif action_char == 'S': i += 1
        elif action_char == 'W': j -= 1

        if (i < 0 or i >= self.height or
            j < 0 or j >= self.width or
            (i, j) == self.block_state):
            return state
        return (i, j)

    def step(self, action_index):
        if self.current_state in self.terminal_states:
            return self.current_state, 0.0, True

        intended_action = self.action_map[action_index]
        transitions = self._get_transition_probs(intended_action)
        possible_actions = list(transitions.keys())
        probs = list(transitions.values())
        actual_action = random.choices(possible_actions, weights=probs, k=1)[0]

        next_state = self._move(self.current_state, actual_action)
        done = next_state in self.terminal_states

        if done:
            reward = self.rewards[next_state]
        else:
            reward = self.living_cost

        self.current_state = next_state
        return next_state, reward, done

# --- DQN agent (inlined for self-contained experiments) ---
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNLearningAgent:
    def __init__(self, env, hyperparameters):
        self.env = env
        self.gamma = hyperparameters['GAMMA']
        self.eps_start = hyperparameters['EPS_START']
        self.eps_end = hyperparameters['EPS_END']
        self.eps_decay = hyperparameters['EPS_DECAY']
        self.learning_rate = hyperparameters['LR']
        self.batch_size = hyperparameters['BATCH_SIZE']
        self.target_update_freq = hyperparameters['TARGET_UPDATE_FREQ']

        self.n_actions = len(env.actions)
        self.n_observations = env.height * env.width

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def _state_to_tensor(self, state):
        row, col = state
        state_index = row * self.env.width + col
        state_tensor = torch.zeros(1, self.n_observations, device=self.device)
        state_tensor[0, state_index] = 1.0
        return state_tensor

    def choose_action(self, state):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                  math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            return torch.tensor([[random.randrange(self.n_actions)]],
                                device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].view(1, 1)

    def learn_from_memory(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)),
                                      device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([
            self._state_to_tensor(s) for s, d in zip(batch.next_state, batch.done) if not d
        ])

        state_batch = torch.cat([self._state_to_tensor(s) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, device=self.device,
                                    dtype=torch.float32)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        target_q_values = reward_batch + (self.gamma * next_q_values)

        criterion = nn.MSELoss()
        loss = criterion(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

# --- Main Training Function ---
def run_training_session(hyperparameters):
    """
    Runs a single, complete training session and returns the rewards.
    (This is the training loop from train.py, but without plots).
    """
    env = GridWorld(trap_penalty=-1.0)
    agent = DQNLearningAgent(env, hyperparameters)
    episode_rewards = []

    num_episodes = hyperparameters['NUM_EPISODES']

    for i_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action_tensor = agent.choose_action(state)
            action = action_tensor.item()
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.memory.push(state, action_tensor, next_state, reward, done)
            state = next_state
            agent.learn_from_memory()

            if done:
                break

        episode_rewards.append(total_reward)

        # Update target network based on the specified frequency
        if i_episode % hyperparameters['TARGET_UPDATE_FREQ'] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    return episode_rewards

def smooth_rewards(rewards, window=100):
    """Applies a moving average to a list of rewards."""
    if len(rewards) < window:
        # Return an empty array if not enough data for one full window
        return np.array([])
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

# --- Hyperparameter Experiments ---

BASE_HYPERPARAMETERS = {
    'NUM_EPISODES': 6000,
    'LR': 0.001,
    'GAMMA': 0.95,
    'BATCH_SIZE': 64,
    'TARGET_UPDATE_FREQ': 10,
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 30000 
}

def run_lr_comparison():
    """Runs training with different Learning Rates and plots the result."""
    print("Running Learning Rate comparison...")
    lr_values = [0.01, 0.001, 0.0001]
    colors = ['red', 'blue', 'green']

    plt.figure(figsize=(12, 6))

    for lr, color in zip(lr_values, colors):
        print(f"--- Training with LR = {lr} ---")
        current_hyperparams = BASE_HYPERPARAMETERS.copy()
        current_hyperparams['LR'] = lr

        rewards = run_training_session(current_hyperparams)
        avg_rewards = smooth_rewards(rewards) # Uses 100-window

        if avg_rewards.size > 0:
            start_index = len(rewards) - len(avg_rewards)
            plt.plot(np.arange(start_index, len(rewards)), avg_rewards,
                     label=f'LR = {lr}', color=color, linewidth=2)
        else:
             print(f"Warning: Not enough episodes ({len(rewards)}) to calculate 100-episode moving average for LR={lr}.")


    plt.title('Effect of Learning Rate (LR) on Reward Convergence (100-ep Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_comparison.png')
    print("Saved 'lr_comparison.png'")
    # plt.show()

def run_gamma_comparison():
    """Runs training with different Gamma values and plots the result."""
    print("\nRunning Gamma (Discount Factor) comparison...")
    gamma_values = [0.3, 0.65, 0.99]
    colors = ['purple', 'orange', 'cyan']

    plt.figure(figsize=(12, 6))

    for gamma, color in zip(gamma_values, colors):
        print(f"--- Training with Gamma = {gamma} ---")
        current_hyperparams = BASE_HYPERPARAMETERS.copy()
        current_hyperparams['GAMMA'] = gamma

        rewards = run_training_session(current_hyperparams)
        avg_rewards = smooth_rewards(rewards) # Uses 100-window

        if avg_rewards.size > 0:
            start_index = len(rewards) - len(avg_rewards)
            plt.plot(np.arange(start_index, len(rewards)), avg_rewards,
                     label=f'Gamma = {gamma}', color=color, linewidth=2)
        else:
            print(f"Warning: Not enough episodes ({len(rewards)}) to calculate 100-episode moving average for Gamma={gamma}.")

    plt.title('Effect of Discount Factor ($gamma$) on Reward Convergence (100-ep Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gamma_comparison.png')
    print("Saved 'gamma_comparison.png'")
    # plt.show()

def run_eps_decay_comparison():
    """Runs training with different Epsilon Decay speeds and plots the result."""
    print("\nRunning Epsilon Decay comparison...")
    # NOTE: These values depend heavily on NUM_EPISODES.
    # For NUM_EPISODES=1000, ~15k steps. 1/3rd = 5k.
    # We'll test: fast (500), medium (5000), slow (15000) - adjust if NUM_EPISODES changes
    eps_decay_values = [500, 20000, 50000] # Adjust if NUM_EPISODES changes significantly
    colors = ['magenta', 'brown', 'lime']

    plt.figure(figsize=(12, 6))

    for decay, color in zip(eps_decay_values, colors):
        print(f"--- Training with EPS_DECAY = {decay} ---")
        current_hyperparams = BASE_HYPERPARAMETERS.copy()
        current_hyperparams['EPS_DECAY'] = decay

        rewards = run_training_session(current_hyperparams)
        avg_rewards = smooth_rewards(rewards) # Uses 100-window

        if avg_rewards.size > 0:
            start_index = len(rewards) - len(avg_rewards)
            plt.plot(np.arange(start_index, len(rewards)), avg_rewards,
                     label=f'EPS_DECAY = {decay}', color=color, linewidth=2)
        else:
            print(f"Warning: Not enough episodes ({len(rewards)}) to calculate 100-episode moving average for EPS_DECAY={decay}.")

    plt.title('Effect of Epsilon Decay Speed on Reward Convergence (100-ep Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('eps_decay_comparison.png')
    print("Saved 'eps_decay_comparison.png'")
    # plt.show()

def run_target_update_comparison():
    """Runs training with different Target Update Frequencies and plots the result."""
    print("\nRunning Target Update Frequency comparison...")
    target_update_values = [1, 10, 100]
    colors = ['navy', 'teal', 'gray']

    plt.figure(figsize=(12, 6))

    for freq, color in zip(target_update_values, colors):
        print(f"--- Training with TARGET_UPDATE_FREQ = {freq} ---")
        current_hyperparams = BASE_HYPERPARAMETERS.copy()
        current_hyperparams['TARGET_UPDATE_FREQ'] = freq

        rewards = run_training_session(current_hyperparams)
        avg_rewards = smooth_rewards(rewards) # Uses 100-window

        if avg_rewards.size > 0:
            start_index = len(rewards) - len(avg_rewards)
            plt.plot(np.arange(start_index, len(rewards)), avg_rewards,
                     label=f'TARGET_UPDATE_FREQ = {freq}', color=color, linewidth=2)
        else:
            print(f"Warning: Not enough episodes ({len(rewards)}) to calculate 100-episode moving average for TARGET_UPDATE_FREQ={freq}.")

    plt.title('Effect of Target Network Update Frequency on Reward Convergence (100-ep Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Moving Avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('target_update_comparison.png')
    print("Saved 'target_update_comparison.png'")
    # plt.show()


if __name__ == "__main__":
    run_lr_comparison()
    run_gamma_comparison()
    run_eps_decay_comparison()
    run_target_update_comparison()
    print("\nAll experiments complete. Plots saved to directory.")