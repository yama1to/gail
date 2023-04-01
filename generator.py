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
            nn.Sigmoid()
        )
        
    def forward(self, state):
        return self.network(state)


class GeneratorRNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(state_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, input):
        output, _ = self.rnn(input)
        output = self.fc(output)
        return output.squeeze(1)



