import torch
import torch.nn as nn
import torch.optim as optim

from generator import GeneratorRNN
from discriminator import DiscriminatorRNN

import gym
from expert_data import load_expert_data


def train_gail(env, expert_data, generator, discriminator, num_epochs=1000, batch_size=1, g_lr=1e-4, d_lr=1e-4):
    expert_states, expert_actions = expert_data
    
    generator_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        idx = torch.randperm(expert_states.size(0))[:batch_size]
        expert_states_batch = expert_states[idx]
        expert_actions_batch = expert_actions[idx]

        # Reshape input tensor to 3-D
        expert_states_batch = expert_states_batch.view(batch_size, -1, generator.state_dim)

        generator_actions_batch = generator(expert_states_batch)
        expert_preds = discriminator(expert_states_batch, expert_actions_batch)
        generator_preds = discriminator(expert_states_batch, generator_actions_batch)

        # Update discriminator
        discriminator_optimizer.zero_grad()
        discriminator_loss = criterion(expert_preds, torch.ones_like(expert_preds)) + \
                             criterion(generator_preds, torch.zeros_like(generator_preds))
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        # Update generator
        generator_optimizer.zero_grad()
        generator_loss = -criterion(discriminator(expert_states_batch, generator_actions_batch), torch.zeros_like(generator_preds))
        generator_loss.backward(retain_graph=True)
        generator_optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Discriminator Loss = {discriminator_loss.item()}, Generator Loss = {generator_loss.item()}')

if __name__ == '__main__':
    
    env = gym.make('Pendulum-v1')
    expert_data = load_expert_data('./prepare/pendulum_expert_data.pkl')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 500

    generator = GeneratorRNN(state_dim, action_dim, hidden_dim)
    discriminator = DiscriminatorRNN(state_dim, action_dim, hidden_dim)

    train_gail(env, expert_data, generator, discriminator) 
