# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

tf.reset_default_graph()
# 创建输入数据
X = np.random.randn(2, 4, 5)  # 批次 、序列长度、样本维度
print(X)

# 第二个样本长度为3
X[1, 2:] = 0
seq_lengths = [4, 2]

Gstacked_rnn = []
Gstacked_bw_rnn = []
for i in range(3):
    Gstacked_rnn.append(tf.contrib.rnn.GRUCell(3))
    Gstacked_bw_rnn.append(tf.contrib.rnn.GRUCell(3))

# 建立前向和后向的三层RNN
Gmcell = tf.contrib.rnn.MultiRNNCell(Gstacked_rnn)
Gmcell_bw = tf.contrib.rnn.MultiRNNCell(Gstacked_bw_rnn)

sGbioutputs, sGoutput_state_fw, sGoutput_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([Gmcell],
                                                                                                   [Gmcell_bw], X,
                                                                                                   sequence_length=seq_lengths,
                                                                                                   dtype=tf.float64)
from tensorflow.python.ops import rnn
Gbioutputs, Goutput_state_fw = rnn.bidirectional_dynamic_rnn(Gmcell, Gmcell_bw, X, sequence_length=seq_lengths,
                                                               dtype=tf.float64)
