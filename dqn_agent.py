import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import namedtuple, deque
import numpy as np

# Define the 5-part memory tuple
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """Experience replay buffer for DQN transitions."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Selects a random batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    Feedforward Q-network for tabular GridWorld states.

    Architecture: 12 (one-hot) -> 64 -> 64 -> 4 (Q-values for N, E, S, W)
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        """Defines the forward pass of the network."""
        x = F.relu(self.layer1(x)) # ReLU activation function
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
        
        # Get number of actions and state observations
        self.n_actions = len(env.actions)
        self.n_observations = env.height * env.width
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy and target networks
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize the optimizer (e.g., Adam)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), 
                                     lr=self.learning_rate, amsgrad=True)
        
        self.memory = ReplayMemory(10000)
        
        self.steps_done = 0

    def _state_to_tensor(self, state):
        """Converts a (row, col) state into a 1x12 one-hot tensor."""
        row, col = state
        state_index = row * self.env.width + col
        # Create a zero tensor
        state_tensor = torch.zeros(1, self.n_observations, device=self.device)
        # Set the one-hot bit
        state_tensor[0, state_index] = 1.0
        return state_tensor
    
    def get_current_epsilon(self):
        """Calculates the current epsilon value based on steps done."""
        # This calculates epsilon based on the total number of steps taken
        return self.eps_end + (self.eps_start - self.eps_end) * \
               math.exp(-1. * self.steps_done / self.eps_decay)

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        # Calculate current epsilon
        epsilon = self.get_current_epsilon()
        self.steps_done += 1
        
        # With prob. epsilon, take a random action (Exploration)
        if random.random() < epsilon:
            return torch.tensor([[random.randrange(self.n_actions)]], 
                                device=self.device, dtype=torch.long)
        else:
            # Otherwise, ask the network for the best action (Exploitation)
            with torch.no_grad(): # We don't need to track gradients here
                state_tensor = self._state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                # Select the action with the highest Q-value
                return q_values.max(1)[1].view(1, 1)

    def learn_from_memory(self):
        """This is the core training step."""
        if len(self.memory) < self.batch_size:
            return # Not enough memories to learn from

        # 1. Sample a random minibatch of memories
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 2. Unpack the batch
        # We need to create tensors for states, actions, rewards, etc.
        
        # Create a mask for non-final (not 'done') states
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), 
                                      device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([
            self._state_to_tensor(s) for s, d in zip(batch.next_state, batch.done) if not d
        ])

        state_batch = torch.cat([self._state_to_tensor(s) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, device=self.device, 
                                    dtype=torch.float32)

        # Q(s, a) from the policy network
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Target: R + gamma * max_a' Q_target(s', a')
        next_q_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute the final target Q-value
        target_q_values = reward_batch + (self.gamma * next_q_values)
        
        # MSE loss and gradient update
        criterion = nn.MSELoss()
        loss = criterion(current_q_values, target_q_values.unsqueeze(1))
        
        # Perform the gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def get_policy(self):
        """Gets the final learned policy from the network."""
        policy = np.full((self.env.height, self.env.width), '', dtype=object)
        for r in range(self.env.height):
            for c in range(self.env.width):
                if (r,c) == self.env.block_state:
                    policy[r,c] = 'X'
                elif (r,c) in self.env.terminal_states:
                    policy[r,c] = 'T'
                else:
                    state_tensor = self._state_to_tensor((r,c))
                    with torch.no_grad():
                        q_values = self.policy_net(state_tensor.to(self.device))
                        best_action_index = q_values.argmax().item()
                        policy[r,c] = self.env.action_map[best_action_index]
        return policy
    
    def get_q_table(self):
        """
        Builds a full 3x4x4 Q-table by querying the policy network
        for every state. This is used for visualization.
        """
        q_table = np.zeros((self.env.height, self.env.width, self.n_actions))
        
        for r in range(self.env.height):
            for c in range(self.env.width):
                state = (r, c)
                if state == self.env.block_state or state in self.env.terminal_states:
                    continue # No Q-values for these states
                
                # Convert state to tensor
                state_tensor = self._state_to_tensor(state).to(self.device)
                
                # Get Q-values from the policy network
                with torch.no_grad():
                    q_values_tensor = self.policy_net(state_tensor)
                
                # Convert to numpy and store in our table
                q_table[r, c, :] = q_values_tensor.cpu().numpy()[0]
                
        return q_table