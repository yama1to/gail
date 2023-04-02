import numpy as np
import pickle
import gym
from stable_baselines3 import PPO



def collect_expert_data(env, model, num_episodes=100, save_path='expert_data.pkl'):
    expert_states = []
    expert_actions = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action, _ = model.predict(state, deterministic=True)
            expert_states.append(state)
            expert_actions.append(action)

            state, _, done, _ = env.step(action)

    expert_states = np.array(expert_states)
    expert_actions = np.array(expert_actions)

    with open(save_path, 'wb') as f:
        pickle.dump((expert_states, expert_actions), f)

    return expert_states, expert_actions

if __name__ == '__main__':
    import argparse

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='1 is Pendulum,\n\
                                                    2 is CartPole-v1')

    # Add an argument
    parser.add_argument('--task', type=str, help='Description of my argument')

    # Parse the arguments
    args = parser.parse_args()

    # Access the value of your argument
    task = int(args.task)

    if task ==1:
        env = gym.make('Pendulum-v1')
        model = PPO.load('pretrained_ppo_pendulum')  # Load a pre-trained agent
        expert_data = collect_expert_data(env, model, num_episodes=200, save_path='pendulum_expert_data.pkl')
        print(f'Expert data saved: {len(expert_data[0])} state-action pairs')

    if task == 2:
        env = gym.make('CartPole-v1')
        model = PPO.load('pretrained_ppo_cartpole')  # Load a pre-trained agent
        expert_data = collect_expert_data(env, model, num_episodes=200, save_path='cartpole_expert_data.pkl')
        print(f'Expert data saved: {len(expert_data[0])} state-action pairs')
