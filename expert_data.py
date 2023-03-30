import numpy as np
import pickle
import torch 

def load_expert_data(file_path):
    with open(file_path, 'rb') as f:
        expert_data = pickle.load(f)

    expert_states, expert_actions = expert_data
    expert_states = torch.tensor(expert_states, dtype=torch.float32)
    expert_actions = torch.tensor(expert_actions, dtype=torch.float32)

    return expert_states, expert_actions

if __name__ == '__main__':
    expert_data_file = 'pendulum_expert_data.pkl'
    expert_states, expert_actions = load_expert_data(expert_data_file)
    print(f'Loaded {len(expert_states)} expert states and {len(expert_actions)} expert actions')
