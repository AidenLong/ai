# -- encoding:utf-8 --
"""
验证码识别（假定验证码中只有：数字、大小写字母，验证码的数目是4个，eg: Gx3f）
过程：
1. 使用训练集进行网络训练
训练集数据怎么来？？===> 使用代码随机的生成一批验证码数据（最好不要每次都随机一个验证码），最好是先随机出10w张验证码的图片，然后利用这10w张图片来训练；否则收敛会特别慢，而且有可能不收敛
如何训练？直接将验证码输入（输入Gx3f），神经网络的最后一层是4个节点，每个节点输出对应位置的值(第一个节点输出：G，第二个节点输出：x，第三个节点输出：3，第四个节点输出：f)
2. 使用测试集对训练好的网络进行测试
3. 当测试的正确率大于75%的时候，模型保存
4. 加载模型，对验证码进行识别
"""

import numpy as np
import matplotlib.pyplot as plt
# captcha是python验证码的库，安装方式pip install captcha
from captcha.image import ImageCaptcha
import random
from PIL import Image
import tensorflow as tf

code_char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'q', 'a', 'z', 'w', 's', 'x', 'e', 'd', 'c', 'r',
                 'f', 'v', 't', 'g', 'b', 'y', 'h', 'n', 'u', 'j',
                 'm', 'i', 'k', 'o', 'l', 'p', 'Q', 'A', 'Z', 'W',
                 'S', 'X', 'E', 'D', 'C', 'R', 'F', 'V', 'T', 'G',
                 'B', 'Y', 'H', 'N', 'U', 'J', 'M', 'I', 'K', 'O',
                 'L', 'P']
code_char_set_size = len(code_char_set)
code_char_2_number_dict = dict(zip(code_char_set, range(len(code_char_set))))
code_number_2_char_dict = dict(zip(range(len(code_char_set)), code_char_set))
# 网络中进行Dropout时候的，神经元的保留率(有多少神经元被保留下来)
# 0.75就表示75%的神经元保留下来，随机删除其中的25%的神经元(其实相当于将神经元的输出值设置为0)
keep_prob = 0.75
# 验证码中的字符数目
code_size = 4


def random_code_text(code_size=4):
    """
    随机产生验证码的字符
    :param code_size:
    :return:
    """
    code_text = []
    for i in range(code_size):
        c = random.choice(code_char_set)
        code_text.append(c)
    return code_text


def generate_code_image(code_size=4):
    """
    产生一个验证码的Image对象
    :param code_size:
    :return:
    """
    image = ImageCaptcha()
    code_text = random_code_text(code_size)
    code_text = ''.join(code_text)
    # 将字符串转换为验证码(流)
    captcha = image.generate(code_text)
    # 如果要保存验证码图片
    # image.write(code_text, 'captcha/' + code_text + '.jpg')

    # 将验证码转换为图片的形式
    code_image = Image.open(captcha)
    code_image = np.array(code_image)

    return code_text, code_image


def code_cnn(x, y):
    """
    构建一个验证码识别的CNN网络
    :param x:  Tensor对象，输入的特征矩阵信息，是一个4维的数据:[number_sample, height, weight, channels]
    :param y:  Tensor对象，输入的预测值信息，是一个2维的数据，其实就是验证码的值[number_sample, code_size]
    :return: 返回一个网络
    """
    """
    网络结构：构建一个简单的CNN网络，因为起始此时验证码图片是一个比较简单的数据，所以不需要使用那么复杂的网络结构，当然了：这种简单的网络结构，80%+的正确率是比较容易的，但是超过80%比较难
    conv -> relu6 -> max_pool -> conv -> relu6 -> max_pool -> dropout -> conv -> relu6 -> max_pool -> full connection -> full connection
    """
    # 获取输入数据的格式，[number_sample, height, weight, channels]
    x_shape = x.get_shape()
    # kernel_size_k: 其实就是卷积核的数目
    kernel_size_1 = 32
    kernel_size_2 = 64
    kernel_size_3 = 64
    unit_number_1 = 1024
    unit_number_2 = code_size * code_char_set_size

    with tf.variable_scope('net', initializer=tf.random_normal_initializer(0, 0.1), dtype=tf.float32):
        with tf.variable_scope('conv1'):
            w = tf.get_variable('w', shape=[5, 5, x_shape[3], kernel_size_1])
            b = tf.get_variable('b', shape=[kernel_size_1])
            net = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)
        with tf.variable_scope('relu1'):
            # relu6和relu的区别：relu6当输入的值大于6的时候，返回6，relu对于大于0的值不进行处理，relu6相对来讲具有一个边界
            # relu: max(0, net)
            # relu6: min(6, max(0, net))
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool1'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('conv2'):
            w = tf.get_variable('w', shape=[3, 3, kernel_size_1, kernel_size_2])
            b = tf.get_variable('b', shape=[kernel_size_2])
            net = tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)
        with tf.variable_scope('relu2'):
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool2'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('dropout1'):
            tf.nn.dropout(net, keep_prob=keep_prob)
        with tf.variable_scope('conv3'):
            w = tf.get_variable('w', shape=[3, 3, kernel_size_2, kernel_size_3])
            b = tf.get_variable('b', shape=[kernel_size_3])
            net = tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)
        with tf.variable_scope('relu3'):
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool3'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('fc1'):
            net_shape = net.get_shape()
            net_sample_feature_number = net_shape[1] * net_shape[2] * net_shape[3]
            net = tf.reshape(net, shape=[-1, net_sample_feature_number])
            w = tf.get_variable('w', shape=[net_sample_feature_number, unit_number_1])
            b = tf.get_variable('b', shape=[unit_number_1])
            net = tf.add(tf.matmul(net, w), b)
        with tf.variable_scope('softmax'):
            w = tf.get_variable('w', shape=[unit_number_1, unit_number_2])
            b = tf.get_variable('b', shape=[unit_number_2])
            net = tf.add(tf.matmul(net, w), b)
    return net


def text_2_vec(text):
    vec = np.zeros((code_size, code_char_set_size))
    k = 0
    for ch in text:
        index = code_char_2_number_dict[ch]
        vec[k][index] = 1
        k += 1
    return np.array(vec.flat)


def vec_2_text(vec):
    text = ''
    index = 0
    for c in vec:
        if c == 1:
            text += code_number_2_char_dict[index % code_char_set_size]
        index += 1
    return text


def random_next_batch(batch_size=64, code_size=4):
    """
    随机获取下一个批次的数据
    :param batch_size:
    :param code_size:
    :return:
    """
    batch_x = []
    batch_y = []

    for i in range(batch_size):
        code, image = generate_code_image(code_size)
        # code字符转换为数字的数组形式
        code_number = text_2_vec(code)
        batch_x.append(image)
        batch_y.append(code_number)

    return np.array(batch_x), np.array(batch_y)


def train_code_cnn(model_path):
    """
    模型训练
    :param model_path:
    :return:
    """
    # 1. 构建相关变量：占位符
    in_image_height = 60
    in_image_weight = 160
    x = tf.placeholder(tf.float32, shape=[None, in_image_height, in_image_weight, 1], name='x')
    y = tf.placeholder(tf.float32, shape=[None, code_size * code_char_set_size], name='y')
    # 1. 获取网络结构
    network = code_cnn(x, y)
    # 2. 构建损失函数（如果四个位置的值，只要有任意一个预测失败，那么我们损失就比较大）
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=y))
    # 3. 定义优化函数
    train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    # 4. 计算准确率
    predict = tf.reshape(network, [-1, code_size, code_char_set_size])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_y = tf.argmax(tf.reshape(y, [-1, code_size, code_char_set_size]), 2)
    correct = tf.equal(max_idx_p, max_idx_y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 5. 开始训练
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # a. 变量的初始化
        sess.run(tf.global_variables_initializer())

        # b. 开始训练
        step = 1
        while True:
            # 1. 获取批次的训练数据
            batch_x, batch_y = random_next_batch(batch_size=64, code_size=code_size)
            # 2. 对数据进行一下处理
            batch_x = tf.image.rgb_to_grayscale(batch_x)
            batch_x = tf.image.resize_images(batch_x, size=(in_image_height, in_image_weight),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # 3. 训练
            _, cost_, accuracy_ = sess.run([train, cost, accuracy], feed_dict={x: batch_x.eval(), y: batch_y})
            print("Step:{}, Cost:{}, Accuracy:{}".format(step, cost_, accuracy_))

            # 4. 每10次输出一次信息
            if step % 10 == 0:
                test_batch_x, test_batch_y = random_next_batch(batch_size=64, code_size=code_size)
                # 2. 对数据进行一下处理
                test_batch_x = tf.image.rgb_to_grayscale(test_batch_x)
                test_batch_x = tf.image.resize_images(test_batch_x, size=(in_image_height, in_image_weight),
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                acc = sess.run(accuracy, feed_dict={x: test_batch_x.eval(), y: test_batch_y})
                print("测试集准确率:{}".format(acc))

                # 如果模型准确率0.7，模型保存，然后退出
                if acc > 0.7 and accuracy_ > 0.7:
                    saver.save(sess, model_path, global_step=step)
                    break

            step += 1


if __name__ == '__main__':
    code_text, code_image = generate_code_image()
    print(code_text)
    print(vec_2_text(text_2_vec(code_text)))
    figure = plt.figure()
    figure.text(0.1, 0.9, code_text)
    plt.imshow(code_image)
    plt.show()
    # train_code_cnn('./model/code/capcha.model')
    # TODO: 作业，假定这个模型运行一段时间后，可以顺利的保存模型；那么自己加入一些代码，代码功能：使用保存好的模型对验证码做一个预测，最终返回值为验证码上的具体字符串； => 下周四，晚自习我讲一下怎么写
    # TODO: 作业，车牌照的识别，分成两个过程：1. 从车辆图片中提取出车牌照区域的值，然后2.对车牌照图片做一个预测； 简化：认为车牌只有蓝牌 ===> 下下周二或者周四，我带着大家写写（大家有一个礼拜的时间自己考虑一下怎么实现）
