import gym
from code.envs.gridworld_env import CliffWalkingWapper
from code.QLearning import agent
from code.QLearning import task0
from code.QLearning import train

env = gym.make('CliffWalking-v0')
env = CliffWalkingWapper(env)

state_dim = env.observation_space
action_dim = env.action_space
state = env.reset()
print(f'状态维度：{state_dim}，动作维度：{action_dim}')
print(f'当前状态：state {state}')
# 状态维度：Discrete(48)，动作维度：Discrete(4)
# 当前状态：state 36

"""RL接口实现"""
env = gym.make('CliffWalking-v0')
env = CliffWalkingWapper(env)
env.seed(1)
state_dim = env.observation_space.n
action_dim = env.action_space
cfg = task0.cfg
agent = agent.QLearning(state_dim, action_dim, cfg)

# for i_ep in range(cfg.train_eps):
#     ep_reward = 0  # 每个epoch的reward
#     state = env.reset()
#     while True:
#         action = agent.choose_action(state)
#         next_state, reward, done, _ = env.step(action)  # 与环境交互得到下一个状态和奖励
#         agent.update(state, action, reward, next_state, done)
#         state = next_state
#         ep_reward += reward
#         if done:
#             break


def train(cfg, env, agent):
    rewards = []
    ma_rewards = []  # 滑动平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            if done:
                break
    rewards.append(ep_reward)
    if ma_rewards:
        ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward*0.1)
    else:
        ma_rewards.append(ep_reward)

print('开始训练')
train(cfg, env, agent)