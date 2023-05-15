## 强化学习入门(paddle)

+ 一、基于表格型方法求解RL
    + MDP、状态价值、Q表格
    + 实践： [Sarsa](01_q_learning)、[Q-learning](02_sarsa)
+ 二、基于神经网络方法求解RL
    + 函数逼近方法
    + 实践：[DQN](03_dqn)
+ 三、基于策略梯度求解RL
    + 策略近似、策略梯度
    + 实践：[Policy Gradient](04_policy_gradient)
+ 四、连续动作空间上求解RL
    + 实战：[DDPG](05_ddpg)

## 使用说明

### 安装依赖（注意：请务必安装对应的版本）

+ [paddlepaddle==1.8.5](https://github.com/PaddlePaddle/Paddle)
+ [parl==1.3.1](https://github.com/PaddlePaddle/PARL)
+ gym

### 运行示例

进入每个示例对应的代码文件夹中，运行
```
python train.py
```
