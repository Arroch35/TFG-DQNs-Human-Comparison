import gymnasium as gym
import numpy as np


# =========================================================
# RESTRICT PONG ACTIONS (NOOP, UP, DOWN)
# =========================================================
class RestrictPongActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.allowed_actions = np.array([0, 2, 3])
        self.action_space = gym.spaces.Discrete(len(self.allowed_actions))

    def action(self, action):
        return int(self.allowed_actions[action])