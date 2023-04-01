import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from generator import GeneratorESN
from discriminator import DiscriminatorESN
from expert_data import load_expert_data
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation


def train_gail(expert_data, generator, discriminator, num_epochs=1000, batch_size=500, 
               g_lr=1e-4, d_lr=1e-4, reservoir_size=1000, spectral_radius=0.9, leaking_rate=0.3):
    expert_states, expert_actions = expert_data
    criterion = nn.BCELoss()

    # ジェネレータとディスクリミネータの最適化アルゴリズムを設定
    generator_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    # ESNパラメータを設定
    generator.set_reservoir_params(reservoir_size=reservoir_size, spectral_radius=spectral_radius, leaking_rate=leaking_rate)
    discriminator.set_reservoir_params(reservoir_size=reservoir_size, spectral_radius=spectral_radius, leaking_rate=leaking_rate)

    # 訓練ループ
    for epoch in range(num_epochs):
        # 訓練データからランダムにバッチをサンプリング
        idx = torch.randperm(expert_states.size(0))[:batch_size]
        expert_states_batch = expert_states[idx]
        expert_actions_batch = expert_actions[idx]

        # ジェネレータによる行動生成
        generator_actions_batch = generator(expert_states_batch)
        # ディスクリミネータによる報酬予測
        expert_preds = discriminator(expert_states_batch, expert_actions_batch)
        generator_preds = discriminator(expert_states_batch, generator_actions_batch)

        # ディスクリミネータの更新
        discriminator_optimizer.zero_grad()
        discriminator_loss = criterion(expert_preds, torch.ones_like(expert_preds)) + \
                             criterion(generator_preds, torch.zeros_like(generator_preds))
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        # ジェネレータの更新
        generator_optimizer.zero_grad()
        generator_loss = -criterion(discriminator(expert_states_batch, generator_actions_batch), torch.zeros_like(generator_preds))
        generator_loss.backward(retain_graph=True)
        generator_optimizer.step()

        # エポックごとに損失を表示
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Discriminator Loss = {discriminator_loss.item()}, Generator Loss = {generator_loss.item()}')

def test_gail(c, generator, reservoir, esn_output_scaling, esn_washout, esn_initial_state_scale):
    # Generatorの性能テスト
    generator.eval()
    env = gym.make(c.gym_task)
    total_reward = test_generator_performance(c, generator, reservoir, esn_output_scaling, esn_washout, esn_initial_state_scale, env, c.test_max_steps)
    print(f'Total reward: {total_reward:.2f}')
    env.close()


def test_generator_performance(c, generator, reservoir, output_scaling, washout, initial_state_scale, env, max_steps=1000):
    state = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    rewardlist = []

    state_data = []
    render_data = []

    # ESNの初期状態を設定
    initial_state = torch.Tensor(np.random.uniform(low=-initial_state_scale, high=initial_state_scale, size=reservoir.n_reservoir)).unsqueeze(0)

    while not done and step < max_steps:
        # ESNの出力を計算
        esn_output = reservoir(torch.Tensor(state).unsqueeze(0), initial_state=initial_state, washout=washout)
        action = generator(esn_output * output_scaling)

        next_state, reward, done, info = env.step(action.detach().numpy())
        state = next_state
        total_reward += reward

        rewardlist.append(reward)
        state_data.append((state, action.item(), reward, done))
        render_data.append(env.render(mode='rgb_array'))

        step += 1

    savefigure(c, rewardlist)
    savegif(c, step, state_data, render_data)
    return total_reward



def getTime():
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") # フォーマットに合わせた現在時刻を取得
    return timestamp

def savefigure(c,rewardlist):
    if c.savefig is False: return
    plt.plot(rewardlist)
    plt.savefig(c.figs)
    plt.cla()

def savegif(c,T,state_data,render_data):
    if c.savegif is False: return
    # 図を初期化
    fig = plt.figure(figsize=(7, 7.5), facecolor='white')
    fig.suptitle(c.gym_task, fontsize=20)
    # print(T,len(state_data),len(render_data))
    # 作図処理を関数として定義
    def update(t):
        # 時刻tの状態を取得
        theta, state, action, reward, terminated = state_data[t]
        rgb_data = render_data[t]
        
        # 状態ラベルを作成
        state_text = 't=' + str(t) + '\n'
        state_text += f'$\\theta$={theta:5.2f}, '
        state_text += f'$\cos(\\theta)$={state[0]:5.2f}, '
        state_text += f'$\\sin(\\theta)$={state[1]:5.2f}\n'
        state_text += f'velocity={state[2]:6.3f}\n'
        if t < T:
            state_text += f'action={action:5.2f}, '
            state_text += f'reward={reward:5.2f}, '
        else:
            state_text += 'action=' + str(action) + ', '
            state_text += 'reward=' + str(reward) + ', '
        state_text += 'terminated:' + str(terminated)
        
        # ペンデュラムを描画
        plt.imshow(rgb_data)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.title(state_text, loc='left')

    # gif画像を作成
    anime = FuncAnimation(fig=fig, func=update, frames=T, interval=100)

    # gif画像を保存
    anime.save(c.gifs, writer='pillow', dpi=100)
    plt.close()


def test_gail(c, generator):
    # Generatorの性能テスト
    generator.eval()
    env = gym.make(c.gym_task)
    total_reward = test_generator_performance(c, generator, env, c.test_max_steps)
    print(f'Total reward: {total_reward:.2f}')
    env.close()



def main(c):
    env = gym.make(c.gym_task)
    expert_data = load_expert_data(c.expert_data)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reservoir_dim = 100

    generator = GeneratorESN(state_dim, action_dim, reservoir_dim)

    discriminator = DiscriminatorESN(state_dim, action_dim, reservoir_dim)

    train_gail(expert_data, generator, discriminator, num_epochs=1000, batch_size=500, 
               g_lr=1e-4, d_lr=1e-4, reservoir_size=1000, spectral_radius=0.9, leaking_rate=0.3)

    test_gail(env, generator)


class Config:
    def __init__(self) -> None:
        self.savefig = True
        self.savegif = True
        self.figs = "./figs/"+getTime()+"reward.png"
        self.gifs = './gifs/'+getTime()+'Pendulum_random.gif'
        self.gym_task = 'Pendulum-v1'
        self.expert_data = './prepare/pendulum_expert_data.pkl'
        self.env = gym.make(self.gym_task)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.hidden_dim = 1000

        self.num_epochs: int = 1000
        self.batch_size: int = 500
        self.g_lr: float = 0.001
        self.d_lr: float = 0.001
        
        self.test_max_steps: int = 1000

if __name__ == '__main__':
    c=Config()
    main(c)