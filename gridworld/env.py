from gym import Env 
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import random

class twoColorsEnv(Env):
    def __init__ (self):
        self.grid_length = 10
        self.grid_width = 10
        self.grid_size = self.grid_length * self.grid_width
        self.subgoals = [self.grid_length - 1] # red (suboptimum)
        self.start = self.grid_size - self.grid_length
        auxiliary = [i for i in range(0, self.grid_size) if i not in self.subgoals]
        self.targets = random.sample(auxiliary, 1) + [self.start] # Blue, agent
        self.action_space = Discrete(4)
        self.state = self.embedding()
        self.observation_space = MultiDiscrete([2] * len(self.state))
        self.needs_reset = False
        self.rewards = [1.]

    def one_hot_encode(self, x, n_classes):
        return np.eye(n_classes)[x]

    def embedding(self):
        pos = [[self.targets[0] // self.grid_length, (self.targets[0] % self.grid_length)], \
               [self.subgoals[0] // self.grid_length, (self.subgoals[0] % self.grid_length)], \
               [self.targets[1] // self.grid_length, (self.targets[1] % self.grid_length)]]
        
        altogether = np.array([])
        for cord in pos:
            altogether = np.append(altogether, self.one_hot_encode(cord, self.grid_length).flatten())
        return altogether

    def rand_loc(self):
        auxiliary = [i for i in range(0, self.grid_size) if i not in self.targets[1:] + self.subgoals]
        self.targets[0] = random.sample(auxiliary, 1)[0]            

    def step(self, action):

        # Actions we can take, up, down, left, right
        if action == 0:
            position = self.targets[-1] - self.grid_length
            if position >= 0:
                self.targets[-1] = position
        elif action == 1:
            position = self.targets[-1] + self.grid_length
            if position <= self.grid_size - 1:
                self.targets[-1] = position        
        elif action == 2:
            if self.targets[-1] % self.grid_length == 0:
                pass
            else:
                self.targets[-1] -= 1 
        elif action == 3:
            if (self.targets[-1] + 1) % self.grid_length == 0:
                pass
            else:                                                   
                self.targets[-1] += 1 
        else:
            raise ValueError('Action invalid!')

        # Calculate reward, observe items' behavior and update state
        if self.targets[-1] == self.targets[0]:
            reward = self.rewards[0]
            self.rand_loc()
            self.state = self.embedding()
            done = True
        elif self.targets[-1] in self.subgoals:
            reward = 0.5
            self.state = self.embedding()
            done = True
            self.needs_reset = True
        else:
            reward = 0
            self.state = self.embedding()
            done = False

        info = {}
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):

        if self.needs_reset:
            self.targets[-1] = self.start
            self.state = self.embedding()
            self.needs_reset = False
            
        return self.state
