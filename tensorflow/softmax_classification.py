# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设置字符集
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 数据加载
mnist = input_data.read_data_sets('data/', one_hot=True)
print(mnist)

# 下载下来的数据集被分三个子集：
# 5.5W行的训练数据集（mnist.train），
# 5千行的验证数据集（mnist.validation)
# 1W行的测试数据集（mnist.test）。
# 因为每张图片为28x28的黑白图片，所以每行为784维的向量。
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print(trainimg.shape)
print(trainlabel.shape)
print(testimg.shape)
print(testlabel.shape)

# 随机展示5张图片
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in randidx:
    # reshape 格式变化
    curr_img = np.reshape(trainimg[i, :], (28, 28))
    curr_label = np.argmax(trainlabel[i, :])
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title('第' + str(i) + '个图，实际数字为：' + str(curr_label))
    plt.show()

# 模型构建
x = tf.placeholder("float", [None, 784], name='x')  # 784是维度（28*28），none表示的是无限多
y = tf.placeholder("float", [None, 10], name='y')

W = tf.Variable(tf.zeros([784, 10]), name='W')  # 每个数字是784像素点的，所以w与x相乘的话也要有784个，b-10表示这个10分类的
b = tf.Variable(tf.zeros([10]), name='b')

actv = tf.nn.softmax(tf.matmul(x, W) + b)
# cost function 均值
# reduce_sum 矩阵和，reduction_indices是指沿tensor的哪些维度求和。
# cost 损失函数
cost = -tf.reduce_mean(tf.reduce_mean(y * tf.log(actv), axis=1))
# 优化
learing_rate = 0.01
# 使用梯度下降，最小化误差
optm = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(cost)

# tf.argmax:对矩阵按行或列计算最大值
# tf.equal:是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# 正确率
accr = tf.reduce_mean(tf.cast(pred, 'float'))
# 初始化
init = tf.global_variables_initializer()

# 总共训练次数
training_epochs = 50
# 批次大小
batch_size = 100
# 训练迭代次数
display_step = 5
# session
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.
    # 5500/100
    num_batch = int(mnist.train.num_examples / batch_size)
    for i in range(num_batch):
        # 获取数据集 next_batch 获取下一批的数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 模型训练
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds) / num_batch

    if epoch % display_step == 0:
        feeds_train = {x: mnist.train.images, y: mnist.train.labels}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print("DONE")
