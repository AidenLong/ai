# -- encoding:utf-8 --
"""
使用TensorFlow训练模型
Create by ibf on 2018/5/28
"""

from plate_number.genplate import GenPlate, gen_sample, chars
import tensorflow as tf
import numpy as np
import os
import cv2


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


def generate_TFRecord(filename, genplate, height=72, weight=272, num_plat=1000):
    """
    随机生成num_plat张车牌照并将数据输出形成TFRecord格式
    :param filename: TFRecord格式文件存储的路径
    :param genplate: 车牌照生成器
    :param height: 车牌照高度
    :param weight: 车牌照宽度
    :param num_plat: 需要生成车牌照的数量
    :return:
    """
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num_plat):
        num, img = gen_sample(genplate, weight, height)
        # TODO: 因为MxNet中的格式要求导致的问题，必须转换回[height, weight, channels]
        img = img.transpose(1, 2, 0)
        img = img.reshape(-1).astype(np.float32)
        num = np.array(num).reshape(-1).astype(np.int32)

        ex = make_example(img.tobytes(), num.tobytes())
        writer.write(ex.SerializeToString())
    writer.close()


def read_tfrecord(filename, x_name='image', y_name='label', x_shape=[72, 272, 1], y_shape=[7], batch_size=64,
                  shuffle_data=False, num_threads=1):
    """
    读取TFRecord文件
    :param filename:
    :param x_name: 给定训练用x的名称
    :param y_name: 给定训练用y的名称
    :param x_shape: x的格式
    :param y_shape: y的格式
    :param batch_size: 批大小
    :param shuffle_data: 是否混淆数据，如果为True，那么进行shuffle操作
    :param num_threads: 线程数目
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
            x_name: tf.FixedLenFeature([], tf.string),
            y_name: tf.FixedLenFeature([], tf.string)
        }
    )

    # 读取特征
    image = tf.decode_raw(features[x_name], tf.float32)
    label = tf.decode_raw(features[y_name], tf.int32)

    # 格式重定
    image = tf.reshape(image, x_shape)
    label = tf.reshape(label, y_shape)

    # 转换为批次的Tensor对象
    capacity = batch_size * 6 + 10
    if shuffle_data:
        image, label = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                              num_threads=num_threads, min_after_dequeue=int(capacity / 2))
    else:
        image, label = tf.train.batch([image, label], batch_size=batch_size, capacity=capacity, num_threads=num_threads)

    return image, label


def model(images, keep_prob):
    """
    模型构建
    :param images: 图像数据，格式：[batch_size,height,width,channels]
    :param keep_prob: 进行dropout时候神经元的保留比例
    :return:
    """

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        net = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        net = tf.nn.relu(tf.nn.bias_add(net, biases))

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 32, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        net = tf.nn.relu(tf.nn.bias_add(net, biases))

    with tf.variable_scope('max_pooling1') as scope:
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 32, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        net = tf.nn.relu(tf.nn.bias_add(net, biases), name=scope.name)

    # conv4
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 64, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[64], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        net = tf.nn.relu(tf.nn.bias_add(net, biases), name=scope.name)

    with tf.variable_scope('max_pooling2') as scope:
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling2')

        # conv5
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 64, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases',
                                 shape=[128], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        net = tf.nn.relu(tf.nn.bias_add(net, biases), name=scope.name)

    # conv6
    with tf.variable_scope('conv6') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 128, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='VALID')
        biases = tf.get_variable('biases',
                                 shape=[128], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        net = tf.nn.relu(tf.nn.bias_add(net, biases), name=scope.name)

    # pool3
    with tf.variable_scope('max_pool3') as scope:
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')

    # fc1_flatten
    with tf.variable_scope('fc1') as scope:
        shp = net.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        net = tf.reshape(net, [-1, flattened_shape])
        net = tf.nn.dropout(net, keep_prob, name='fc1_dropdot')

    with tf.variable_scope('fc21') as scope:
        weights = tf.get_variable('weights',
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1)
                                 )
        net1 = tf.matmul(net, weights) + biases
    with tf.variable_scope('fc22') as scope:
        weights = tf.get_variable('weights',
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1)
                                 )
        net2 = tf.matmul(net, weights) + biases
    with tf.variable_scope('fc23') as scope:
        weights = tf.get_variable('weights',
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1)
                                 )
        net3 = tf.matmul(net, weights) + biases
    with tf.variable_scope('fc24') as scope:
        weights = tf.get_variable('weights',
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1)
                                 )
        net4 = tf.matmul(net, weights) + biases
    with tf.variable_scope('fc25') as scope:
        weights = tf.get_variable('weights',
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1)
                                 )
        net5 = tf.matmul(net, weights) + biases
    with tf.variable_scope('fc26') as scope:
        weights = tf.get_variable('weights',
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1)
                                 )
        net6 = tf.matmul(net, weights) + biases
    with tf.variable_scope('fc27') as scope:
        weights = tf.get_variable('weights',
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1)
                                 )
        net7 = tf.matmul(net, weights) + biases

    # shape = [7,batch_size,65]
    return net1, net2, net3, net4, net5, net6, net7


def losses(logits1, logits2, logits3, logits4, logits5, logits6, logits7, labels):
    """
    基于给定的预测值和目标属性，构建损失函数
    :param logits1: [batch_size, 65]
    :param logits2: [batch_size, 65]
    :param logits3: [batch_size, 65]
    :param logits4: [batch_size, 65]
    :param logits5: [batch_size, 65]
    :param logits6: [batch_size, 65]
    :param logits7: [batch_size, 65]
    :param labels: [batch_size, 7]
    :return:
    """
    labels = tf.convert_to_tensor(labels, tf.int32)

    with tf.variable_scope('loss1') as scope:
        # tf.nn.softmax_cross_entropy_with_logits：要求logits和labels格式必须一样，是：[batch_sise, class_number]
        # tf.nn.sparse_softmax_cross_entropy_with_logits：把softmax_cross_entropy_with_logits简化了，labels中不需给定一个二维的数组形式，只需要一个[batch_size]，中间的每个元素是实际样本的对应目标属性的index下标
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels[:, 0])
        loss1 = tf.reduce_mean(cross_entropy, name='loss1')
        tf.summary.scalar(scope.name + '/loss1', loss1)

    with tf.variable_scope('loss2') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels[:, 1])
        loss2 = tf.reduce_mean(cross_entropy, name='loss2')
        tf.summary.scalar(scope.name + '/loss2', loss2)

    with tf.variable_scope('loss3') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3, labels=labels[:, 2])
        loss3 = tf.reduce_mean(cross_entropy, name='loss3')
        tf.summary.scalar(scope.name + '/loss3', loss3)

    with tf.variable_scope('loss4') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits4, labels=labels[:, 3])
        loss4 = tf.reduce_mean(cross_entropy, name='loss4')
        tf.summary.scalar(scope.name + '/loss4', loss4)

    with tf.variable_scope('loss5') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits5, labels=labels[:, 4])
        loss5 = tf.reduce_mean(cross_entropy, name='loss5')
        tf.summary.scalar(scope.name + '/loss5', loss5)

    with tf.variable_scope('loss6') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits6, labels=labels[:, 5])
        loss6 = tf.reduce_mean(cross_entropy, name='loss6')
        tf.summary.scalar(scope.name + '/loss6', loss6)

    with tf.variable_scope('loss7') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits7, labels=labels[:, 6])
        loss7 = tf.reduce_mean(cross_entropy, name='loss7')
        tf.summary.scalar(scope.name + '/loss7', loss7)

    return loss1, loss2, loss3, loss4, loss5, loss6, loss7


def create_optimizer(loss1, loss2, loss3, loss4, loss5, loss6, loss7, learning_rate):
    """
    基于目标函数构建优化器
    :param loss1:
    :param loss2:
    :param loss3:
    :param loss4:
    :param loss5:
    :param loss6:
    :param loss7:
    :param learning_rate:
    :return:
    """
    with tf.name_scope('optimizer1'):
        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op1 = optimizer1.minimize(loss1, global_step=global_step)
    with tf.name_scope('optimizer2'):
        optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op2 = optimizer2.minimize(loss2, global_step=global_step)
    with tf.name_scope('optimizer3'):
        optimizer3 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op3 = optimizer3.minimize(loss3, global_step=global_step)
    with tf.name_scope('optimizer4'):
        optimizer4 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op4 = optimizer4.minimize(loss4, global_step=global_step)
    with tf.name_scope('optimizer5'):
        optimizer5 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op5 = optimizer5.minimize(loss5, global_step=global_step)
    with tf.name_scope('optimizer6'):
        optimizer6 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op6 = optimizer6.minimize(loss6, global_step=global_step)
    with tf.name_scope('optimizer7'):
        optimizer7 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op7 = optimizer7.minimize(loss7, global_step=global_step)

    return train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7


def evaluation(logits1, logits2, logits3, logits4, logits5, logits6, logits7, labels):
    """
    计算准确率，返回表示准确率的tensor对象
    :param logits1: [batch_size, 65]
    :param logits2: [batch_size, 65]
    :param logits3: [batch_size, 65]
    :param logits4: [batch_size, 65]
    :param logits5: [batch_size, 65]
    :param logits6: [batch_size, 65]
    :param logits7: [batch_size, 65]
    :param labels: [batch_size, 7]
    :return:
    """
    # 按照行对数据进行组合
    logits_all = tf.concat([logits1, logits2, logits3, logits4, logits5, logits6, logits7], 0)
    # 格式转换为：[batch_size, 7](先显示第一个字符的所有样本，在显示第二个.... => 和logists_all类似)
    labels = tf.convert_to_tensor(labels, tf.int32)
    labels_all = tf.reshape(tf.transpose(labels), [-1])
    # 计算准确率
    with tf.variable_scope('accuracy') as scope:
        # in_top_k(predictions, targets, k, name=None)
        """
        in_top_k(predictions, targets, k, name=None)
            predictions： 预测值，格式为：N*M
            targets：最大目标索引值， 格式为：N*k
            k：数字k
        API含义：判断predictions中，和targets对应行中的对应targets中的value值是不是predictions中对应行中所有M个数据的最大K个数
        """
        # 具体有那一些是最大值
        correct = tf.nn.in_top_k(logits_all, labels_all, 1)
        # 类型从bool转换为float
        correct = tf.cast(correct, tf.float16)
        # 求准确率
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


def train(train_plat_tfrecord_fname, img_h=72, img_w=272, channels=3, num_label=7, batch_size=8, max_epoch=30000,
          learning_rate=0.00001, logs_train_dir='./models'):
    """
    模型训练
    :return:
    """
    # 读取TFRecord文件Tensor对象
    train_image, train_label = read_tfrecord(train_plat_tfrecord_fname, x_shape=[img_h, img_w, channels],
                                             y_shape=[num_label], batch_size=batch_size, shuffle_data=True)

    # 定义输入的占位符
    keep_prob = tf.placeholder(tf.float32)

    # 获取得到模型/获取得到预测值
    logits1, logits2, logits3, logits4, logits5, logits6, logits7 = model(train_image, keep_prob)

    # 定义构建损失函数
    loss1, loss2, loss3, loss4, loss5, loss6, loss7 = losses(logits1, logits2, logits3,
                                                             logits4, logits5, logits6,
                                                             logits7, train_label)

    # 定义构建优化器
    op1, op2, op3, op4, op5, op6, op7 = create_optimizer(loss1, loss2, loss3, loss4, loss5, loss6, loss7, learning_rate)

    # 计算准确率
    train_acc = evaluation(logits1, logits2, logits3, logits4, logits5, logits6, logits7, train_label)

    # 合并所有输出
    summary_op = tf.summary.merge_all()

    # 启动会话
    with tf.Session() as sess:
        # 构建训练可视化日志
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        # 构建模型保存对象
        saver = tf.train.Saver()

        # 启动线程组
        # 初始化全局变量
        init_op = tf.group(
            tf.local_variables_initializer(),  # 可以注释，但是有的tensorflow版本中，必须写这一行
            tf.global_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 遍历迭代训练
        for step in range(max_epoch):
            feed_dict = {keep_prob: 0.5}
            _, _, _, _, _, _, _, loss1_, loss2_, loss3_, loss4_, loss5_, loss6_, loss7_, train_acc_, summary_str = sess.run(
                [
                    op1, op2, op3, op4, op5, op6, op7,
                    loss1, loss2, loss3, loss4, loss5, loss6, loss7,
                    train_acc, summary_op],
                feed_dict
            )
            train_writer.add_summary(summary_str, step)

            if step % 10 == 0:
                all_loss = loss1_ + loss2_ + loss3_ + loss4_ + loss5_ + loss6_ + loss7_
                print('Step %d,train_loss = %.2f,acc= %.2f' % (step, all_loss, train_acc_))

                if (step % 10 == 0 and step != 0) or (step + 1) == max_epoch:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


def predict(image, logs_train_dir='./models', img_h=72, img_w=272, channels=3):
    x = tf.placeholder(tf.float32, [None, img_h, img_w, channels])
    keep_prob = tf.placeholder(tf.float32)

    logits1, logits2, logits3, logits4, logits5, logits6, logits7 = model(x, keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Reading checkpoint...")
        chkpt_fname = tf.train.latest_checkpoint(logs_train_dir)
        if chkpt_fname:
            print("Load checkpoint...")
            saver.restore(sess, chkpt_fname)

        # 得到预测信息
        pre1, pre2, pre3, pre4, pre5, pre6, pre7 = sess.run(
            [logits1, logits2, logits3, logits4, logits5, logits6, logits7],
            feed_dict={x: image, keep_prob: 1.0})

        prediction = np.reshape(np.array([pre1, pre2, pre3, pre4, pre5, pre6, pre7]), [-1, 65])
        prediction_index = []
        prediction_line = ''
        for i in range(prediction.shape[0]):
            if i == 0:
                index = np.argmax(prediction[i][0:31])
            if i == 1:
                index = np.argmax(prediction[i][41:65]) + 41
            if i > 1:
                index = np.argmax(prediction[i][31:65]) + 31

            prediction_index.append(index)
            prediction_line += chars[index] + " "

    return prediction_index, prediction_line


if __name__ == '__main__':
    op = 1
    filename = './plate_train.tfrecord'
    genplate = GenPlate("./font/platech.ttf", './font/platechar.ttf', './NoPlates')
    height = 72
    weight = 272
    train_plate_number = 100

    if op == 1:
        # 样本数据产生
        generate_TFRecord(filename, genplate, height, weight, num_plat=train_plate_number)
    elif op == 2:
        # 模型训练
        train(filename, img_h=height, img_w=weight, channels=3, num_label=7, batch_size=12, max_epoch=30000,
              learning_rate=0.0001, logs_train_dir='./models')
    else:
        label, img = gen_sample(genplate, weight, height)
        # TODO: 因为MaxNet中的格式要求导致的问题，必须转换回[height, weight, channels]
        img = img.transpose(1, 2, 0)
        # 预测
        print("实际索引为：{}".format(label))
        index, value = predict([img], logs_train_dir='./models', img_h=height, img_w=weight, channels=3)
        print("预测索引为：{}".format(index))
        print("预测车牌号为:{}".format(value))
        # 展示一下
        cv2.imshow('plate', img)
        cv2.waitKey(0)
