# -*- coding:utf-8 -*-

import tensorflow as tf


def get_variable(name, shape=None, dtype=tf.float32, initialize=tf.random_normal_initializer(mean=0, stddev=0.1)):
    """
    返回一个对应的变量
    :param name:
    :param shape:
    :param dtype:
    :param initialize:
    :return:
    """
    return tf.get_variable(name, shape, dtype, initialize)


def vgg_a(x):
    net1_kernel_size = 64
    net3_kernel_size = 128
    net5_kernel_size = 256
    net7_kernel_size = 512
    net9_kernel_size = 512
    net11_unit_size = 4096
    net12_unit_size = 4096
    net13_unit_size = 1000
    with tf.variable_scope('net1'):
        net = tf.nn.conv2d(x, filter=get_variable('W', [3, 3, 3, net1_kernel_size]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net1_kernel_size]))
        net = tf.nn.relu(net)
    with tf.variable_scope('net2'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    with tf.variable_scope('net3'):
        net = tf.nn.conv2d(net, filter=get_variable('w', [3, 3, net1_kernel_size, net3_kernel_size]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net3_kernel_size]))
        net = tf.nn.relu(net)
    with tf.variable_scope('net4'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    with tf.variable_scope('net5'):
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net3_kernel_size, net5_kernel_size]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net5_kernel_size]))
        net = tf.nn.relu(net)
        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net5_kernel_size, net5_kernel_size]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net5_kernel_size]))
        net = tf.nn.relu(net)
    with tf.variable_scope('net6'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    with tf.variable_scope('net7'):
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net5_kernel_size, net7_kernel_size]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net7_kernel_size]))
        net = tf.nn.relu(net)
        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net7_kernel_size, net7_kernel_size]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net7_kernel_size]))
        net = tf.nn.relu(net)
    with tf.variable_scope('net8'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    with tf.variable_scope('net9'):
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net7_kernel_size, net9_kernel_size]),
                           strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net9_kernel_size]))
        net = tf.nn.relu(net)
        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net9_kernel_size, net9_kernel_size]),
                           strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net9_kernel_size]))
        net = tf.nn.relu(net)
    with tf.variable_scope('net10'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    with tf.variable_scope('net11'):
        shape = net.get_shape()
        feature_number = shape[1] * shape [2] * shape[3]
        net = tf.reshape(net, [-1, feature_number])
        # 全连接
        net = tf.add(tf.matmul(net, get_variable('w', [feature_number, net11_unit_size])), get_variable('b', [net11_unit_size]))
    with tf.variable_scope('net12'):
        # 全连接
        net = tf.add(tf.matmul(net, get_variable('w', [net11_unit_size, net12_unit_size])), get_variable('b', [net12_unit_size]))
    with tf.variable_scope('net13'):
        # 全连接
        net = tf.add(tf.matmul(net, get_variable('w', [net12_unit_size, net13_unit_size])),
                     get_variable('b', [net13_unit_size]))
    with tf.variable_scope('net14'):
        # softmax
        act = tf.nn.softmax(net)
    return act
