# -- encoding:utf-8 --
"""
TensorFlow官方推荐使用TFRecord作为数据的格式化存储工具，可以提高TensorFlow训练过程中的IO效率<br/>
TFRecord内部使用“Protocol Buffer”二进制数据编码方案，只要生成一次TFRecord，之后的数据读取、加工处理的效率都比较高<br/>
TensorFlow中数据读入一般四种方式：
    1. 预先把所有数据加载内存
        对内存压力比较大，容易出现OOM
    2. 在每轮训练中使用原生Python代码读取一部分数据，然后使用feed_dict输入到计算图
        如果需要数据重复读取的时候，IO操作比较麻烦
    3. 利用Threading和Queues从TFRecord中分批次读取数据
    4. 使用Dataset API
        内部用TFRecord，一般指使用TensorFlow默认数据集
使用TFRecord的时候，数据单位一般是：tf.train.Example或者tf.train.SequenceExample
    tf.train.Example:
        一般用于数值、图像等固定大小的数据，使用tf.train.Feature指定每个记录各特征的名称和数据类型
        使用方式：
            tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Features(int64_list=tf.train.Int64List(value=[height])),
                'weight': tf.train.Features(int64_list=tf.train.Int64List(value=[width])),
                'depth': tf.train.Features(int64_list=tf.train.Int64List(value=[depth])),
                'image': tf.train.Features(bytes_list=tf.train.BytesList(value=[image])),
            }))
            备注：Features支持三种数据：int64_list、float_list、bytes_list
    tf.train.SequenceExample:
        一般用于文本、时间序列等没有固定长度大小的数据
        使用方式：
            example = tf.train.SequenceExample()
            # 通过context来制定数据量大小
            example.context.feature['length'].int64_list.value.append(len(words))

            # 通过feature_lists来加载数据
            word_lists = example.feature_lists.feature_list['word']
            for word in words:
                word_lists.feature_add().int64_list.value.append(word_id(word))
Create by ibf on 2018/5/29
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 1. TFRecord生成
def make_example(image, label):
    """
    产生Example对象
    :param image:
    :param label:
    :return:
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))


def write_tfrecord(images, labels, filename):
    """
    将数据输出成为TFRecord格式文件
    :param images:
    :param labels:
    :param filename:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(filename)
    for image, label in zip(images, labels):
        label = label.astype(np.float32)
        image = image.astype(np.float32)

        ex = make_example(image.tobytes(), label.tobytes())
        writer.write(ex.SerializeToString())
    writer.close()


def writer(print_write_test=True):
    """
    测试写数据形成TFRecord文件
    :return:
    """
    mnist = input_data.read_data_sets('./data', one_hot=True)
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    # 写出TFRecord
    write_tfrecord(train_images, train_labels, 'mnist_train.tfrecord')
    write_tfrecord(test_images, test_labels, 'mnist_test.tfrecord')

    # 查看一条数据
    if print_write_test:
        test_example = next(tf.python_io.tf_record_iterator("mnist_test.tfrecord"))
        print(tf.train.Example.FromString(test_example))


def read_tfrecord(filename):
    """
    读取TFRecord文件
    :param filename:
    :return:
    """
    # 获取队列
    filename_queue = tf.train.string_input_producer([filename])
    # 构建数据读取器
    reader = tf.TFRecordReader()
    # 读取队列中的数据
    _, serialized_example = reader.read(filename_queue)

    # 处理样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }
    )

    # 读取特征
    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)

    # 格式重定
    image = tf.reshape(image, [28, 28, 1])
    label = tf.reshape(label, [10])

    # 转换为批次的Tensor对象
    image, label = tf.train.batch([image, label], batch_size=64, capacity=500)

    return image, label


def get_variable(name, shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    """
    返回一个对应的变量
    :param name:
    :param shape:
    :param dtype:
    :param initializer:
    :return:
    """
    return tf.get_variable(name, shape, dtype, initializer)


def mnist_model(image, label):
    """
    构建模型
    :param image:
    :param label:
    :return:
    """
    # 1. 模型构建
    net = tf.nn.conv2d(input=image, filter=get_variable('w1', [5, 5, 1, 20]), strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, get_variable('b1', [20])))
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    net = tf.nn.conv2d(input=net, filter=get_variable('w2', [5, 5, 20, 50]), strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, get_variable('b2', [50])))
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    size = 7 * 7 * 50
    net = tf.reshape(net, shape=[-1, size])
    net = tf.add(tf.matmul(net, get_variable('w3', [size, 500])), get_variable('b3', [500]))
    net = tf.nn.relu(net)

    net = tf.add(tf.matmul(net, get_variable('w4', [500, 10])), get_variable('b4', [10]))

    # 构建损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=label))

    # 构建优化器
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    return train, loss


def train_mnist_model():
    """
    构建一个基于TFRecord的手写数字识别
    :return:
    """
    # 读取TFRecord文件Tensor对象
    train_image, train_label = read_tfrecord('mnist_train.tfrecord')
    print(train_image)

    # 构建返回的训练器
    train, loss = mnist_model(train_image, train_label)

    step = 0
    with tf.Session() as sess:
        init_op = tf.group(
            # tf.local_variables_initializer(), # 可以注释，但是有的tensorflow版本中，必须写这一行
            tf.global_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while step < 2000:
            # 模型训练
            _, loss_ = sess.run([train, loss])
            if step % 10 == 0:
                print("step:{}, loss:{}".format(step, loss_))
            step += 1
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    writer()
    # print(read_tfrecord('mnist_train.tfrecord'))
    train_mnist_model()
