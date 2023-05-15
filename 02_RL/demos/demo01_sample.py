import gym
import numpy as np

if __name__ == '__main__':
    # 环境1：FrozenLake, 可以配置冰面是否是滑的
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up

    # 环境2：CliffWalking, 悬崖环境
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

    # PARL代码库中的`examples/tutorials/lesson1`中`gridworld.py`提供了自定义格子世界的封装，可以自定义配置格子世界的地图

    env.reset()
    for step in range(100):
        action = np.random.randint(0, 4)
        obs, reward, done, info = env.step(action)
        print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(step, action, obs, reward, done, info))
        env.render()
