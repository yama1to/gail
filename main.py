import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

from generator import Generator 
from discriminator import Discriminator



# GAILの学習アルゴリズム
def train_gail(env_name, expert_traj, expert_policy, generator, discriminator, num_epochs, batch_size, expert_traj_len):
    # 環境の初期化
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_dim = 256

    # モデルの初期化
    optimizer_g = optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-3)

    # GAILの学習
    for epoch in range(num_epochs):
        # 生成器の学習
        obs = env.reset()
        for t in range(expert_traj_len):
            # 生成器による行動の生成
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_tensor = generator(obs_tensor).squeeze(0)
                action = action_tensor.numpy()
            
            # 環境における実行
            next_obs, reward, done, info = env.step(action)
            expert_traj.append((obs, action, reward, next_obs, done))
            obs = next_obs

            if done:
                obs = env.reset()

        # 識別器の学習
        for _ in range(batch_size):
            # expertのサンプルの取得
            expert_obs, expert_act, _, _, _ = expert_traj[np.random.randint(len(expert_traj))]

            # generatorのサンプルの取得
            obs_tensor = torch.tensor(expert_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                expert_act_tensor = torch.tensor(expert_act, dtype=torch.float32).unsqueeze(0)
                generated_act_tensor = generator(obs_tensor).squeeze(0)
            expert_score = discriminator(torch.cat([obs_tensor, expert_act_tensor], dim=1))
            generated_score = discriminator(torch.cat([obs_tensor, generated_act_tensor], dim=1))
            loss_d = -torch.mean(torch.log(expert_score) + torch.log(1 - generated_score))
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        # 生成器の学習
        for _ in range(batch_size):
            # expertのサンプルの取得
            expert_obs, expert_act, _, _, _ = expert_traj[np.random.randint(len(expert_traj))]

            # generatorのサンプルの取得
            obs_tensor = torch.tensor(expert_obs, dtype=torch.float32).unsqueeze(0)
            generated_act_tensor = generator(obs_tensor).squeeze(0)
            generated_score = discriminator(torch.cat([obs_tensor, generated_act_tensor], dim=1))
            loss_g = -torch.mean(torch.log(generated_score))

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        # 1エポックごとに結果を表示
        print(f"Epoch {epoch}: loss_d={loss_d.item()}, loss_g={loss_g.item()}")


    # 結果の出力
    obs = env.reset()
    rewards = []
    for t in range(1000):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_tensor = generator(obs_tensor).squeeze(0)
            action = action_tensor.numpy()

        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        rewards.append(reward)
        if done:
            obs = env.reset()

    print(f"Average reward: {np.mean(rewards)}")


if __name__ == "__main__":
    env_name = "NineRooms-v0"

    obs_dim, act_dim, hidden_dim = 8,3, 100

    expert_traj = []
    expert_policy = lambda obs: np.array([0, 1, 0])  
    
    generator = Generator(obs_dim, act_dim, hidden_dim)
    discriminator = Discriminator(obs_dim, act_dim, hidden_dim)

    train_gail(env_name, expert_traj, expert_policy, generator, 
            discriminator, num_epochs=50, batch_size=32, expert_traj_len=100)
