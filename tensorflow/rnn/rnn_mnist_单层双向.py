# -*- coding:utf-8 -*-
"""
使用RNN实现手写数字识别的功能
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 数据加载(每个样本都是784维的)
mnist = input_data.read_data_sets('data/', one_hot=True)

# 构建一个会话
with tf.Session() as sess:
    lr = 0.001  # 学习率
    # 每个时刻输入的数据维度大小
    input_size = 28
    # 时刻数目，总共输入多少次
    timestep_size = 28
    # 细胞中一个神经网络的层次中的神经元的数目
    hidden_size = 128
    # RNN中的隐层的数目
    layer_num = 2
    # 最后输出的类别数目
    class_num = 10

    _X = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, class_num])
    # batch_size是一个int32类型的标量tensor的占位符，使用batch_size可以让我们在训练和测试的时候使用不同的数据量
    batch_size = tf.placeholder(tf.int32, [])
    # dropout的时候，保留率多少
    keep_prod = tf.placeholder(tf.float32, [])

    # 开始网络构建
    # 1. 输入的数据格式转换
    # X格式：[batch_size, time_steps, input_size]
    X = tf.reshape(_X, shape=[-1, timestep_size, input_size])

    # 单层LSTM RNN
    # 2. 定义cell
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, reuse=tf.get_variable_scope().reuse)
    gru_cell_bw = tf.nn.rnn_cell.GRUCell(num_units=hidden_size, reuse=tf.get_variable_scope().reuse)

    # 3. 单层的RNN网络应用
    init_state_fw = lstm_cell_fw.zero_state(batch_size, dtype=tf.float32)
    init_state_bw = gru_cell_bw.zero_state(batch_size, dtype=tf.float32)
    # 3. 动态构建双向的RNN网络
    """
    bidirectional_dynamic_rnn(
        cell_fw: 前向的rnn cell
        , cell_bw：反向的rnn cell
        , inputs：输入的序列
        , sequence_length=None
        , initial_state_fw=None：前向rnn_cell的初始状态
        , initial_state_bw=None：反向rnn_cell的初始状态
        , dtype=None
        , parallel_iterations=None
        , swap_memory=False, time_major=False, scope=None)
    API返回值：(outputs, output_states) => outputs存储网络的输出信息，output_states存储网络的细胞状态信息
    outputs: 是一个二元组, (output_fw, output_bw)构成，output_fw对应前向的rnn_cell的执行结果，结构为：[batch_size, time_steps, output_size];output_bw对应反向的rnn_cell的执行结果，结果和output_bw一样
    output_states：是一个二元组，(output_state_fw, output_state_bw) 构成，output_state_fw和output_state_bw是dynamic_rnn API输出的状态值信息
    """
    # time_major=False,默认就是False，output的格式为：[batch_size, timestep_size, hidden_size],
    # 获取最后一个时刻的输出值是：output_ = output[:,-1,:] 一般就是默认值
    outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=gru_cell_bw, inputs=X,
                                                     initial_state_bw=init_state_bw, initial_state_fw=init_state_fw)
    output_fw = outputs[0][:, -1, :]
    output_bw = outputs[0][:, -1, :]
    output = tf.concat([output_fw, output_bw], 1)

    # 将输出值（最后一个时刻对应的输出值构建为全连接）
    w = tf.Variable(tf.truncated_normal([hidden_size * 2, class_num], mean=0.0, stddev=0.1), dtype=tf.float32,
                    name='out_w')
    b = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32, name='out_b')
    y_pre = tf.nn.softmax(tf.matmul(output, w) + b)

    # 损失函数定义
    loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_pre), 1))
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # 准确率
    cp = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(cp, tf.float32))

    # 开始训练
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)
        # 训练模型
        sess.run(train, feed_dict={_X: batch[0], y: batch[1], keep_prod: 0.5, batch_size: _batch_size})
        # 隔一段时间计算准确率
        if (i + 1) % 20 == 0:
            train_acc = sess.run(accuracy,
                                 feed_dict={_X: batch[0], y: batch[1], keep_prod: 1.0, batch_size: _batch_size})
            print("批次:{}, 步骤:{}, 训练集准确率:{}".format(mnist.train.epochs_completed, (i + 1), train_acc))

    # 测试集准确率计算
    test_acc = sess.run(accuracy, feed_dict={_X: mnist.test.images, y: mnist.test.labels, keep_prod: 1.0,
                                             batch_size: mnist.test.num_examples})
    print("测试集准确率:{}".format(test_acc))
