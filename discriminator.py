import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return self.network(state_action)