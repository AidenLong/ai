# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

# 构造数据
np.random.seed(28)
N = 100
x = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)
y = 14 * x - 7 + np.random.normal(loc=0.0, scale=1.0, size=N)

# 将x和y设置为矩阵
x.shape = -1, 1
y.shape = -1, 1
# print(x)

# 模型构建 y = wx + b
# 定义一个变量w和b
w = tf.Variable(initial_value=tf.random_uniform([1], -1.0, 1.0), name='w')
b = tf.Variable(initial_value=tf.zeros([1]), name='b')
# 构建一个预测值
y_hat = w * x + b

# 构建一个损失函数 mse
loss = tf.reduce_mean(tf.square(y_hat - y), name='loss')

# 以随机梯度下降的方式优化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
# 在优化过程中，是让那个函数最小化
train = optimizer.minimize(loss, name='train')

# 全局变量跟新
init_op = tf.global_variables_initializer()


def print_info(r_w, r_b, r_loss):
    print('w={}, b={}, loss={}'.format(r_w, r_b, r_loss))


# 运行
with tf.Session() as sess:
    sess.run(init_op)

    r_w, r_b, r_loss = sess.run([w, b, loss])
    print_info(r_w, r_b, r_loss)

    # 训练（20次）
    for step in range(200):
        sess.run(train)
        r_w, r_b, r_loss = sess.run([w, b, loss])
        print_info(r_w, r_b, r_loss)
