import gymnasium as gym
import numpy as np
from engine import MineSweeper
from config import LEVEL


rows, cols, mines = LEVEL



class MineEnv(gym.Env):
    def __init__(self, safe_first = False):
        self.max_steps = rows * cols
        self.steps = 0
        self.reward_per_open = 1/(rows*cols-mines)
        self.action_space = gym.spaces.Discrete(3*rows*cols)
        self.observation_space = gym.spaces.Box(
            low=0, high = 1,
            shape=(11, rows, cols),
            dtype=np.float32
        )
        self.game = MineSweeper(LEVEL, safe_first)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)       
        self.game.rng = self.np_random
        self.game.reset()
        self.steps = 0
        return self._obs(), {}
    
    def action_masks(self):
        mask = np.zeros((3, rows, cols), dtype=bool)
        mask[0] = self.game.board == -2
        return mask.reshape(-1)
        
            
    
    def step(self, action):
        hidden_before = self.game.hidden
        info = {}
    
        n = rows*cols
        kind, cell = divmod(int(action), n)
        r, c = divmod(cell, cols)
        # guess = no revealed neighbours before opening (first click excluded)
        guess = self.steps != 0 and np.count_nonzero(self.game.board[max(r-1, 0):min(r+2, rows), max(c-1, 0):min(c+2, cols)] > -2) == 0

        if kind == 0:
            safe = self.game.open(c, r)
        else:
            safe = True
        opened = hidden_before - self.game.hidden
        if not safe:
            reward, terminated = -1.0, True
        elif self.game.win:
            reward, terminated = 1.0, True
        else:
            reward, terminated = 0.0, False
        
        if opened:         
            # reward += opened*self.reward_per_open
            reward += 0.12
        elif not terminated:
            reward -= 0.1

        if guess:
            reward -= 0.3

        self.steps += 1
        truncated = self.steps >= self.max_steps
        
        if terminated or truncated:
            info["is_success"] = self.game.win
        
        return self._obs(), reward, terminated, truncated, info


    def _obs(self):
        obs = (self.game.board[None, :, :] == np.arange(-2, 9)[:, None, None]).astype(np.float32)
        return obs