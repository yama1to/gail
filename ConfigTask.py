import gym 

def getTime():
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") # フォーマットに合わせた現在時刻を取得
    return timestamp

class ConfigTask:
    def __init__(self):
        return 
    def set_Pendulum(self):
        self.gifs = './gifs/'+getTime()+'Pendulum_random.gif'
        self.gym_task = 'Pendulum-v1'
        self.expert_data = './prepare/pendulum_expert_data.pkl'

        self.env = gym.make(self.gym_task)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]


    def set_CartPole(self):
        """
        state : dim4 , the cart position, cart velocity, pole angle, and pole angular velocity.
        action : dim1 , possible actions
        """
        self.gifs = './gifs/'+getTime()+'CartPole_random.gif'
        self.gym_task = 'CartPole-v1'
        self.expert_data = './prepare/cartpole_expert_data.pkl'

        self.env = gym.make(self.gym_task)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 1

    def set_MountainCar(self):

        self.gifs = './gifs/'+getTime()+'MountainCar_random.gif'
        self.gym_task = 'MountainCar-v0'
        self.expert_data = './prepare/mountaincar_expert_data.pkl'

        self.env = gym.make(self.gym_task)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 1
