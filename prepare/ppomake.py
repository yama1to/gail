from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

# Create the environment
env = gym.make('Pendulum-v1')
env = DummyVecEnv([lambda: env])

# Train the model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model to a file
model.save('pretrained_ppo_pendulum.zip')
