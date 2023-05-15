import numpy as np


class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n  # 动作数量
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # 奖励的衰减率
        self.epsilon = e_greed  # 按一定概率随机选择动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, done):
        """
        off-policy
        :param obs: 交互前的obs, s_t
        :param action: 本次交互选择的action, a_t
        :param reward: 本次动作获得的奖励r
        :param next_obs: 本次交互后的obs, s_t+1
        :param done: episode是否结束
        :return:
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    # 把 Q表格 的数据保存到文件中
    def save(self, npy_file="./q_table.npy"):
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到 Q表格
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
