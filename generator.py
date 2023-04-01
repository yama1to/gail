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
    


def create_weight_matrix(input_size, hidden_size, spectral_radius, connection_prob):
    w = torch.randn(hidden_size, input_size)

    # Apply connection probability
    w *= (torch.rand_like(w) < connection_prob).float()

    # Scale weight matrices by spectral radius
    if input_size == hidden_size:
        w *= spectral_radius / torch.max(torch.abs(torch.linalg.eig(w)[0]))

    else :
        w*= spectral_radius

    return w

class GeneratorESN(nn.Module):
    def __init__(self, state_dim, action_dim, reservoir_dim,alpha_i,alpha_r,beta_i,beta_r):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reservoir_dim = reservoir_dim
        self.reset_states()

        self.win = nn.Parameter(create_weight_matrix(self.state_dim, self.reservoir_dim, alpha_i, beta_i))
        self.w =  nn.Parameter(create_weight_matrix(self.reservoir_dim, self.reservoir_dim, alpha_r, beta_r))
        # self.wfb = nn.Parameter(torch.zeros((action_dim, reservoir_dim)))
        nn.init.uniform_(self.w, -1, 1)
        # nn.init.uniform_(self.wfb, -1, 1)
        self.wo = nn.Parameter(torch.zeros((action_dim, reservoir_dim)))
        nn.init.uniform_(self.wo, -1, 1)

        self.rho = 0.5

        self.activation = nn.Tanh()

    def reset_states(self):
        self.x = torch.zeros((1, self.reservoir_dim))

    def forward(self, input_data):
        with torch.no_grad():
            x_new = self.activation(torch.mm(input_data, self.win.t()) + torch.mm(self.x, self.w))
            self.x = (1 - self.rho) * self.x + self.rho * x_new
        y = torch.mm(self.x, self.wo.t())
        # print(tuple(input_data.shape),tuple(self.x.shape), tuple(self.wo.t().shape))
        return y



