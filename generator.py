import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np



# 生成器の定義
class Generator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.model(obs)