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


class GeneratorRNN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(GeneratorRNN, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(obs_dim + act_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, obs, actions):
        h = torch.zeros(1, obs.size(0), self.hidden_dim)
        c = torch.zeros(1, obs.size(0), self.hidden_dim)
        for t in range(obs.size(1)):
            obs_t = obs[:, t, :].unsqueeze(1)
            action_t = actions[:, t, :].unsqueeze(1)
            x = torch.cat([obs_t, action_t], dim=2)
            _, (h, c) = self.rnn(x, (h, c))

        return self.fc(h.squeeze(0))
