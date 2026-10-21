import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class FeatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 256):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
            
            
        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observation):
        output = self.cnn(observation)
        return self.linear(output)
    
    
    
if __name__ == '__main__':
    from minesweeper_env import MineEnv
    space = MineEnv().observation_space
    net = FeatureCNN(space)
    out = net(torch.zeros(1, *space.shape))
    print(out.shape)