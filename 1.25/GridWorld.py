import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.table import Table

# 格子尺寸
WORLD_SIZE = 5
# 状态A的位置(下标从0开始,下同)
A_POS = [0, 1]
# 状态A'的位置
A_PRIME_POS = [4, 1]
# 状态B的位置
B_POS = [0, 3]
# 状态B'的位置
B_PRIME_POS = [2, 3]
# 折扣因子
DISCOUNT = 0.9

# 动作集={上,下,左,右}
ACTIONS = [np.array([-1, 0]),
           np.array([1, 0]),
           np.array([0, 1]),
           np.array([0, -1]),
]
# 策略,每个动作等概率
ACTION_PROB = 0.25

def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # 添加表格
    for (i,j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # 行标签
    for i, label in enumerate(range(len(image))):
        tb.add_cell(i, -1, width, height, text=label+1, loc='right',
                    edgecolor='none', facecolor='none')
    # 列标签
    for j, label in enumerate(range(len(image))):
        tb.add_cell(WORLD_SIZE, j, width, height/2, text=label+1, loc='center',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)


def step(state, action):
    '''给定当前状态以及采取的动作,返回后继状态及其立即奖励

    Parameters
    ----------
    state : list
        当前状态
    action : list
        采取的动作

    Returns
    -------
    tuple
        后继状态,立即奖励

    '''
    # 如果当前位置为状态A,则直接跳到状态A',奖励为+10
    if state == A_POS:
        return A_PRIME_POS, 10
    # 如果当前位置为状态B,则直接跳到状态B',奖励为+5
    if state == B_POS:
        return B_PRIME_POS, 5

    state = np.array(state)
    # 通过坐标运算得到后继状态
    next_state = (state + action).tolist()
    x, y = next_state
    # 根据后继状态的坐标判断是否出界
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        # 出界则待在原地,奖励为-1
        reward = -1.0
        next_state = state
    else:
        # 未出界则奖励为0
        reward = 0
    return next_state, reward

def bellman_equation():
    ''' 求解贝尔曼(期望)方程'''
    # 初始值函数
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        new_value = np.zeros(value.shape)
        # 遍历所有状态
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                # 遍历所有动作
                for action in ACTIONS:
                    # 执行动作,转移到后继状态,并获得立即奖励
                    (next_i, next_j), reward = step([i, j], action)
                    # 贝尔曼期望方程
                    new_value[i, j] += ACTION_PROB * \
                    (reward + DISCOUNT * value[next_i, next_j])
        # 迭代终止条件: 误差小于1e-4
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.title('$v_{\pi}$')
            plt.show()
            plt.close()
            break
        value = new_value

def bellman_optimal_equation():
    '''求解贝尔曼最优方程'''
    # 初始值函数
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        new_value = np.zeros(value.shape)
        # 遍历所有状态
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                values = []
                # 遍历所有动作
                for action in ACTIONS:
                    # 执行动作,转移到后继状态,并获得立即奖励
                    (next_i, next_j), reward = step([i, j], action)
                    # 缓存动作值函数 q(s,a) = r + γ*v(s')
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                # 根据贝尔曼最优方程,找出最大的动作值函数 q(s,a) 进行更新
                new_value[i, j] = np.max(values)
        # 迭代终止条件: 误差小于1e-4
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.title('$v_{*}$')
            plt.show()
            plt.close()
            break
        value = new_value

bellman_equation()
bellman_optimal_equation()