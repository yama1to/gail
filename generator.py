import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np



class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.network(state)
