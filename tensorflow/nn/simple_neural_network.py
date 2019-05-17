# -*- coding:utf-8 -*-
# 引入包
import tensorflow as tf
import matplotlib as mpl
from tensorflow.examples.tutorials.mnist import input_data

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 数据加载
mnist = input_data.read_data_sets('data/', one_hot=True)

# 构建神经网络(4层、1 input, 2 hidden，1 output)
n_unit_hidden_1 = 256  # 第一层hidden中的神经元数目
n_unit_hidden_2 = 128  # 第二层的hidden中的神经元数目
n_input = 784  # 输入的一个样本（图像）是28*28像素的
n_classes = 10  # 输出的类别数目

# 定义输入的占位符
x = tf.placeholder(tf.float32, shape=(None, n_input))
y = tf.placeholder(tf.float32, shape=(None, n_classes))

# 初始化的w和b
weights = {
    # stddev 标准差
    'w1': tf.Variable(tf.random_normal(shape=[n_input, n_unit_hidden_1], stddev=0.1)),
    'w2': tf.Variable(tf.random_normal(shape=[n_unit_hidden_1, n_unit_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.random_normal(shape=[n_unit_hidden_2, n_classes], stddev=0.1))
}
biases = {
    # stddev 标准差
    'b1': tf.Variable(tf.random_normal(shape=[n_unit_hidden_1], stddev=0.1)),
    'b2': tf.Variable(tf.random_normal(shape=[n_unit_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.random_normal(shape=[n_classes], stddev=0.1))
}


def multiplayer_perceotron(_X, _weights, _biases):
    with tf.variable_scope('layer1'):
        # 第一层 -> 第二层 input -> hidden1
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']), name='layer1')
    with tf.variable_scope('layer2'):
        # 第二层 -> 第三层 hidden1 -> hidden2
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2']), name='layer2')
    with tf.variable_scope('layer3'):
        # 第三层 -> 第四层 hidden2 -> output
        layer3 = tf.add(tf.matmul(layer2, _weights['out']), _biases['out'], name='layer3')
    return layer3


# 获取预测值
act = multiplayer_perceotron(x, weights, biases)

# 构建模型的损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=y))

# 使用梯度下降求解，最小化误差
# learning_rate
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 得到预测的类别是哪一个
pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
# 正确率,(True转换为1，False转换为1)
acc = tf.reduce_mean(tf.cast(pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 执行模型的训练
batch_size = 100  # 每次处理的图片数
display_step = 4  # 每4次迭代打印一次

with tf.Session() as sess:
    sess.run(init)

    # 模型保存、持久化
    saver = tf.train.Saver()
    epoch = 0
    while True:
        avg_cost = 0
        # 计算出总的批次
        total_batch = int(mnist.train.num_examples / batch_size)
        # 迭代更新
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            sess.run(train, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)

        avg_cost = avg_cost / total_batch

        if (epoch + 1) % display_step == 0:
            print("批次: %03d 损失函数值: %.9f" % (epoch, avg_cost))
            feeds = {x: mnist.train.images, y: mnist.train.labels}
            train_acc = sess.run(acc, feed_dict=feeds)
            print("训练集准确率: %.3f" % train_acc)
            feeds = {x: mnist.test.images, y: mnist.test.labels}
            test_acc = sess.run(acc, feed_dict=feeds)
            print("测试准确率: %.3f" % test_acc)

            if train_acc > 0.9 and test_acc > 0.9:
                saver.save(sess, './mn/model')
                break

        epoch += 1

    writer = tf.summary.FileWriter('./mn/graph', tf.get_default_graph())
    writer.close()
