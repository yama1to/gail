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
               g_lr=1e-4, d_lr=1e-4):
    expert_states, expert_actions = expert_data
    criterion = nn.BCELoss()
    torch.autograd.set_detect_anomaly(True)
    # ジェネレータとディスクリミネータの最適化アルゴリズムを設定
    generator_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    # 訓練ループ
    for epoch in range(num_epochs):
        # 訓練データからランダムにバッチをサンプリング
        idx = torch.randperm(expert_states.size(0))[:batch_size]
        expert_states_batch = expert_states[idx]
        expert_actions_batch = expert_actions[idx].view(batch_size, c.action_dim)

        # ジェネレータによる行動生成
        generator_actions_batch = generator(expert_states_batch)

        # print(expert_actions_batch.shape, generator_actions_batch.shape)
        # print(expert_actions_batch.shape,expert_actions_batch[0],generator_actions_batch[0])
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
        # print(generator_loss)
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


def test_generator_performance(c, generator, env, max_steps=1000):
    state = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    rewardlist = []

    state_data = []
    render_data = [] # 最初の状態
    # print(state.shape)
    generator.reset_states()
    while not done and step < max_steps:
        theta = env.state[0].copy()
        action = generator(torch.Tensor(state).view(1,c.state_dim))

        if c.gym_task=='CartPole-v1':
            toEnvAct = action.squeeze().detach().numpy()
            toEnvAct = torch.round(torch.sigmoid(toEnvAct)).int()
        if c.gym_task == 'MountainCar-v0':
            toEnvAct = action.squeeze().detach().numpy()
            
            if toEnvAct < -0.33:
                toEnvAct = 0
            elif toEnvAct < 0.33:
                toEnvAct = 1
            elif 0.33 <= toEnvAct :
                toEnvAct = 2
            
        else:
            toEnvAct = action.squeeze(0).detach().numpy()

        next_state, reward, done, info = env.step(toEnvAct)
        state = next_state
        total_reward += reward
        # print(action.shape)

        rewardlist.append(reward)
        state_data.append((theta, state, toEnvAct, reward, done)) # 現在
        if c.render is True:
            render_data.append(env.render(mode='rgb_array')) # 次
        step += 1

    savefigure(c,rewardlist)
    savegif(c,step,state_data,render_data)
    return total_reward,step




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
        
        # print(state.shape)
        # 状態ラベルを作成
        state_text = ''
        if c.gym_task == 'Pendulum-v1':
            state_text += 't=' + str(t) + '\n'
            state_text += f'$\\theta$={float(theta):5.2f}, '
            state_text += f'$\cos(\\theta)$={float(state[0]):5.2f}, '
            state_text += f'$\\sin(\\theta)$={float(state[1]):5.2f}\n'
            state_text += f'velocity={float(state[2]):6.3f}\n'
            if t < T:
                state_text += f'action={action:5.2f}, '
                state_text += f'reward={float(reward):5.2f}, '
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
    anime = FuncAnimation(fig=fig, func=update, frames=T, interval=200,cache_frame_data=False)

    # gif画像を保存
    anime.save(c.gifs, writer='pillow', dpi=100)
    plt.close()

def test_gail(c, generator):
    # Generatorの性能テスト
    generator.eval()
    env = gym.make(c.gym_task)
    steps = 0 
    total_rewards = 0
    goal = 0

    testtimes = 100
    for i in range(testtimes):
        total_reward,step = test_generator_performance(c, generator, env, c.test_max_steps)
        total_rewards += total_reward
        steps += step
        if total_reward > -200: goal += 1

    print(f'Total reward: {total_rewards:.2f}')
    print(f'Total goal: {goal}/{testtimes}')
    env.close()

def main(c):
    expert_data = load_expert_data(c.expert_data)

    generator = GeneratorESN(c.state_dim, c.action_dim, c.hidden_dim,
                             alpha_i=0.1,alpha_r=0.9,beta_i=0.9,beta_r=0.05)

    discriminator = DiscriminatorESN(c.state_dim, c.action_dim, c.hidden_dim,
                                     alpha_i=0.8,alpha_r=0.9,beta_i=0.8,beta_r=0.1)

    train_gail(expert_data, generator, discriminator, num_epochs=c.num_epochs,
                batch_size=c.batch_size, g_lr=c.g_lr, d_lr=c.g_lr)

    test_gail(c, generator)


class Config:
    def __init__(self) -> None:
        self.savefig = False
        self.savegif = False
        self.render = False
        self.figs = "./figs/"+getTime()+"reward.png"

        from ConfigTask import ConfigTask as taskcfg
        task = taskcfg()
        # task.set_CartPole()
        # task.set_Pendulum()
        task.set_MountainCar()

        self.gifs = task.gifs
        self.gym_task = task.gym_task
        self.expert_data = task.expert_data

        self.env = task.env

        self.state_dim = task.state_dim
        self.action_dim = task.action_dim
        self.hidden_dim = 500

        # print(self.state_dim, self.action_dim, self.hidden_dim)

        self.num_epochs: int = 100000
        self.batch_size: int = 64
        self.g_lr: float = 1e-4
        self.d_lr: float = 1e-4

        self.test_max_steps: int = 1000

if __name__ == '__main__':
    c=Config()
    main(c)

