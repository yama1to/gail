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


class DiscriminatorRNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DiscriminatorRNN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        batch_size = state.shape[0]
        seq_len = state.shape[1]
        h0 = torch.zeros(1, batch_size, self.hidden_dim)

        x = torch.cat([state, action], dim=2)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out
