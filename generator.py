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

class GeneratorESN(nn.Module):
    def __init__(self, state_dim, action_dim, reservoir_dim):
        super().__init__(state_dim, action_dim, reservoir_dim)
        self.reset_states()

    def set_reservoir_params(self, reservoir_size, spectral_radius, leaking_rate):
        super().set_reservoir_params(reservoir_size=reservoir_size, spectral_radius=spectral_radius, leaking_rate=leaking_rate)
    
    def reset_states(self):
        self.x = torch.zeros((1, self.reservoir_size))
        
    def forward(self, input_data):
        input_data = input_data.view(input_data.size(0), -1)
        
        x_new = self.activation(torch.mm(input_data, self.win.t()) + torch.mm(self.x, self.w) \
                                + torch.mm(self.prev_output, self.wfb.t()))
        self.x = (1 - self.rho) * self.x + self.rho * x_new
        
        self.prev_output = x_new.mm(self.wfb)
        
        return self.prev_output


