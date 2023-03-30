import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np


# 識別器の定義
class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act):
        obs_act = torch.cat([obs, act], dim=1)
        return self.model(obs_act)