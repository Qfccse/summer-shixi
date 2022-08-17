import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation


# 神经网络
class Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Net, self).__init__()
        # 这里使用两个全连接层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 隐藏层使用relu激活函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# 经验回放池
class ExpReplay(object):
    def __init__(self, capacity):
        # 定义回放池最大容量
        self.buffer = collections.deque(maxlen=capacity)

    # 放入回放池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 随机从回放池中采样
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update):
        self.action_dim = action_dim

        # 评估网络
        self.q_eval = Net(state_dim, hidden_dim, self.action_dim)
        # 目标网络
        self.q_target = Net(state_dim, hidden_dim,
                            self.action_dim)
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action = self.q_eval(state).argmax().item()

        return action

    def update(self, transition):
        # 获取小车的四个状态以及游戏是否结束的信息
        states = torch.FloatTensor(transition[0])
        actions = torch.tensor(transition[1]).view(-1, 1)
        rewards = torch.FloatTensor(transition[2]).view(-1, 1)
        next_states = torch.FloatTensor(transition[3])
        dones = torch.FloatTensor(transition[4]).view(-1, 1)

        # 计算Q值
        q_values = self.q_eval(states).gather(1, actions)
        # 计算下一步最大的Q值
        next_q_values = self.q_target(next_states).max(1)[0].view(-1, 1)
        # 累计Q值
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.q_target.load_state_dict(
                self.q_eval.state_dict())  # 更新目标网络
        self.count += 1


lr = 2e-3
num_episodes = 500  # 进行500局游戏
hidden_dim = 128
gamma = 0.98  # 折扣因子
epsilon = 0.01  # epsilon-贪婪策略
target_update = 10  # 更新频率
buffer_size = 10000  # 经验池容量
minimal_size = 200  # 开始学习时经验池的容量
batch_size = 64

env = gym.make('CartPole-v0')
replay_buffer = ExpReplay(buffer_size)
state_num = env.observation_space.shape[0]
action_num = env.action_space.n
agent = DQN(state_num, hidden_dim, action_num, lr, gamma, epsilon, target_update)


return_list = []
a = 1
for i in range(10):
    for i_episode in range(int(num_episodes / 10)):
        big_reward = 0
        episode_return = 0
        state = env.reset()
        done = False
        env.render()
        while not done:
            big_reward += 1
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            episode_return += reward
            state = next_state
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition = [b_s, b_a, b_r, b_ns, b_d]
                agent.update(transition)
        return_list.append(episode_return)
        print(episode_return)

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.show()
