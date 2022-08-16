from gym import Env 
from gym.spaces import Discrete, Box
import numpy as np
import random

class DiagonalEnv(Env):
    def __init__ (self):
        self.grid_length = 10
        self.grid_width = 10
        self.grid_size = self.grid_length * self.grid_width
        self.subgoal = self.grid_length - 1                   # Subgoal at the top right corner
        self.optimum = self.grid_size - self.grid_length      # Optimum at the bottom left corner
        self.start = 0
        self.agent_pos = self.start                           # Initial agent position
        self.action_space = Discrete(4)
        self.state = self.embedding()
        self.observation_space = Box(
                low=0, high=1, shape=(len(self.state),), dtype=np.float64
            ) # For vectorization, we use Box instead of MultiDiscrete
        
    def one_hot_encode(self, x, n_classes):
        return np.eye(n_classes)[x]

    def embedding(self):
        pos = [[self.optimum // self.grid_length, (self.optimum % self.grid_length)], \
               [self.subgoal // self.grid_length, (self.subgoal % self.grid_length)], \
               [self.agent_pos // self.grid_length, (self.agent_pos % self.grid_length)]]
        
        altogether = np.array([])
        for cord in pos:
            altogether = np.append(altogether, self.one_hot_encode(cord, self.grid_length).flatten())
        return altogether

    def step(self, action):

        # Actions we can take, up, down, left, right
        if action == 0:
            position = self.agent_pos - self.grid_length
            if position >= 0:
                self.agent_pos = position
        elif action == 1:
            position = self.agent_pos + self.grid_length
            if position <= self.grid_size - 1:
                self.agent_pos = position        
        elif action == 2:
            if self.agent_pos % self.grid_length == 0:
                pass
            else:
                self.agent_pos -= 1 
        elif action == 3:
            if (self.agent_pos + 1) % self.grid_length == 0:
                pass
            else:                                                   
                self.agent_pos += 1 
        else:
            raise ValueError('Action invalid!')

        # Calculate reward, observe items' behavior and update state
        if self.agent_pos == self.optimum:
            reward = 5
            done = True
        elif self.agent_pos == self.subgoal:
            reward = 4.5
            done = True
        else:
            reward = 0
            done = False

        self.state = self.embedding()            
        info = {}
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self, return_info=False):

        self.agent_pos = self.start
        self.state = self.embedding()

        if not return_info:
            return self.state
        else:
            return self.state, {}
        
