# -*- coding:utf-8 -*-
import numpy as np
import hmmlearn.hmm as hmm

# 定义变量
states = ['盒子1', '盒子2', '盒子3']
obs = ['白球', '黑球']
n_states = len(states)
m_obs = len(obs)
start_probability = np.array([0.2, 0.5, 0.3])
transition_probability = np.array([
    [0.5, 0.4, 0.1],
    [0.2, 0.2, 0.6],
    [0.2, 0.5, 0.3]
])
emission_probalitity = np.array([
    [0.4, 0.6],
    [0.8, 0.2],
    [0.5, 0.5]
])

# 定义模型
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probalitity

# 运行viterbi预测的问题
se = np.array([[0, 1, 0, 0, 1]]).T
logprod, box_index = model.decode(se, algorithm='viterbi')
print("颜色:", end="")
print(" ".join(map(lambda t: obs[t], [0, 1, 0, 0, 1])))
print("盒子:", end="")
print(" ".join(map(lambda t: states[t], box_index)))
print("概率值:", end="")
print(np.exp(logprod))  # 这个是因为在hmmlearn底层将概率进行了对数化，防止出现乘积为0的情况
