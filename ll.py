import gym
from time import sleep
import numpy as np

class LunarLander:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.env.render()

    def do_action(self, action):
        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    def reset(self):
        return self.env.reset()

    def render(self, timestep=1./60):
        self.env.render()
        sleep(timestep)
