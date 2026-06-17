import numpy as np
import random

class GridWorld:
    """
    A 3x4 stochastic GridWorld environment with a Gym-like interface.

    Grid coordinates (row, col), 0-indexed from top-left:
    (0,0) (0,1) (0,2) (0,3) [GOAL, +1]
    (1,0) (1,1) [BLOCK] (1,3) [TRAP, -1]
    (2,0) [START] (2,1) (2,2) (2,3)
    """
    def __init__(self, trap_penalty=-1.0):
        self.height = 3
        self.width = 4
        
        # Define grid elements
        self.start_state = (2, 0)
        self.goal_state = (0, 3)
        self.trap_state = (1, 3)
        self.block_state = (1, 1)
        self.terminal_states = [self.goal_state, self.trap_state]

        self.rewards = np.zeros((self.height, self.width))
        self.rewards[self.goal_state] = 1.0
        self.rewards[self.trap_state] = trap_penalty
        self.living_cost = -0.04
        
        # Actions: 0=N, 1=E, 2=S, 3=W
        self.actions = ['N', 'E', 'S', 'W']
        self.action_map = { 0: 'N', 1: 'E', 2: 'S', 3: 'W' }
        
        # State: current (row, col)
        self.current_state = self.start_state

    def reset(self):
        """Resets the environment to the start state."""
        self.current_state = self.start_state
        return self.current_state

    def _get_transition_probs(self, action_char):
        """
        Gets the stochastic transition probabilities.
        80% intended, 10% left, 10% right.
        """
        idx = self.actions.index(action_char)
        left = self.actions[(idx - 1) % 4]
        right = self.actions[(idx + 1) % 4]
        return {action_char: 0.8, left: 0.1, right: 0.1}

    def _move(self, state, action_char):
        """
        Moves from a state given an action, checking for walls/blocks.
        """
        i, j = state
        if action_char == 'N':
            i -= 1
        elif action_char == 'E':
            j += 1
        elif action_char == 'S':
            i += 1
        elif action_char == 'W':
            j -= 1
            
        # Check for collision with walls or block
        if (i < 0 or i >= self.height or 
            j < 0 or j >= self.width or 
            (i, j) == self.block_state):
            return state
        
        return (i, j)

    def step(self, action_index):
        """
        Takes one step in the environment.
        
        Args:
            action_index (int): 0=N, 1=E, 2=S, 3=W

        Returns:
            tuple: (next_state, reward, done)
        """
        if self.current_state in self.terminal_states:
            # Should not step from a terminal state, but good to handle
            return self.current_state, 0.0, True

        # Get the intended action character
        intended_action = self.action_map[action_index]
        
        # Get the transition probabilities
        transitions = self._get_transition_probs(intended_action)
        
        # Choose the actual action based on stochastic probabilities
        possible_actions = list(transitions.keys())
        probs = list(transitions.values())
        actual_action = random.choices(possible_actions, weights=probs, k=1)[0]
        
        # Get the next state
        next_state = self._move(self.current_state, actual_action)
        
        # Check if the next state is terminal
        done = next_state in self.terminal_states
        
        # Get the reward
        if done:
            reward = self.rewards[next_state]
        else:
            reward = self.living_cost
            
        # Update the environment's state
        self.current_state = next_state
        
        return next_state, reward, done