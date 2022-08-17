import torch  # 导入torch
import torch.nn as nn  # 导入torch.nn
import torch.nn.functional as F  # 导入torch.nn.functional
import numpy as np  # 导入numpy
import gym  # 导入gym
import matplotlib.pyplot as plt

# 超参数
batch_size = 32  # 样本数量
lr = 0.01  # 学习率
epsilon = 0.9  # greedy policy
gamma = 0.9  # reward discount
memo_iter = 100  # 目标网络更新频率
memo_size = 2000  # 记忆库容量
env = gym.make('CartPole-v0').unwrapped  # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
a_num = env.action_space.n  # 杆子动作个数 (2个)
s_num = env.observation_space.shape[0]  # 杆子状态个数 (4个)


# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):  # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()  # 等价与nn.Module.__init__()
        self.fc1 = nn.Linear(s_num, 50)  # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(50, a_num)  # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):  # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))  # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out(x)  # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value  # 返回动作值


class ReplayMemory(object):
    def __init__(self):
        self.memory_counter = 0
        self.memory = np.zeros((memo_size, s_num * 2 + 2))

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % memo_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample(self):
        sample_index = np.random.choice(memo_size, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :s_num])
        b_a = torch.LongTensor(b_memory[:, s_num:s_num + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, s_num + 1:s_num + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -s_num:])
        return b_s, b_a, b_r, b_s_


relay_memory = ReplayMemory()


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def take_action(self, state):  # 定义动作选择函数 (x为状态)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < epsilon:  # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            action = torch.max(self.eval_net(state), 1)[1].data.numpy()[0]  # 输出每一行最大值的索引，并转化为numpy ndarray形式
        else:  # 随机选择动作
            action = np.random.randint(0, a_num)  # 这里action随机等于0或1 (N_ACTIONS = 2)

        return action

    def update(self):  # 定义学习函数(记忆库已满后便开始学习)
        if self.step_counter % memo_iter == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.step_counter += 1  # 学习步数自加1
        b_s, b_a, b_r, b_s_ = relay_memory.sample()
        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()  # 令dqn=DQN类

reward_list = []

for i in range(300):  # 400个episode循环
    print('Episode: %s' % i)
    s = env.reset()  # 重置环境
    reward = 0  # 初始化该循环对应的episode的总奖励

    while True:  # 开始一个episode (每一个循环代表一步)
        # env.render()                                                    # 显示实验动画
        a = dqn.take_action(s)  # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)  # 执行动作，获得反馈

        # 修改奖励使模型更快的收敛
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        relay_memory.store_transition(s, a, new_r, s_)  # 存储样本
        reward += new_r  # 逐步加上一个episode内每个step的reward
        s = s_  # 更新状态
        if relay_memory.memory_counter > memo_size:  # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.update()

        if done:  # 如果done为True
            print('episode%s---reward_sum: %s' % (i, reward))
            reward_list.append(reward)
            break  # 该episode结束

plt.plot(range(len(reward_list)), reward_list)
plt.show()
