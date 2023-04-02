from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

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

if task == 1:
    name = 'Pendulum-v1'
    fname = 'pretrained_ppo_pendulum.zip'
if task == 2:
    name = 'CartPole-v1'
    fname = 'pretrained_ppo_cartpole.zip'

if task == 3 : 
    name = 'MountainCar-v0'
    fname = 'pretrained_ppo_mountaincar.zip'

# Create the environment
env = gym.make(name)
env = DummyVecEnv([lambda: env])

# Train the model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Save the model to a file
model.save(fname)
