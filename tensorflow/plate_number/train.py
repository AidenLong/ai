# --encoding:utf-8 --
"""
模型训练，并将模型训练结果进行输出
"""

import mxnet as mx
import cv2
from plate_number.genplate import *


class OCRBatch(object):
    """进行模型训练的过程中，当前批次的数据表示对象"""

    def __init__(self, data_names, data, label_names, label):
        """
        构造函数
        data_names: 图片名称
        data: 图片数据
        label_names: 标签名称
        label: 标签数据
        """
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class OCRIter(mx.io.DataIter):
    """构建的一个专门用于MxNet网络的数据输入迭代器"""

    def __init__(self, count, batch_size, num_label, height, width):
        """
        初始化方法
        count => 数据量
        batch_size => 指定每个批次使用多少照片进行模型训练
        num_label => label的数量, 车牌预测中是7个
        height => 给定图片的高度
        width => 给定图片的宽度
        """
        super(OCRIter, self).__init__()
        self.genplate = GenPlate("./font/platech.ttf", './font/platechar.ttf', './NoPlates')
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]

    def __iter__(self):
        for k in range((int)(self.count / self.batch_size)):
            # 进行第k次的模型训练
            data = []
            label = []
            # 进行当前批次的数据获取
            for i in range(self.batch_size):
                num, img = gen_sample(self.genplate, self.width, self.height)
                data.append(img)
                label.append(num)

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass


def get_ocrnet():
    # 1. 设置输出的变量x和y名称
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')

    # 2. 网络构建-第一部分
    # 2.1. 卷积层
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
    # 2.2. 池化层
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(1, 1))
    # 2.3. 激活层
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    # 3. 网络构建-第二部分
    # 3.1. 卷积层
    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), num_filter=32)
    # 3.2. 池化层
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    # 3.3. 激活层
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    # 4. 网络构建-全连接层构建
    flatten = mx.symbol.Flatten(data=relu2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=120)
    fc21 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc22 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc24 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc25 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc26 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc27 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25, fc26, fc27], dim=0)
    # label转置
    label = mx.symbol.transpose(data=label)
    # label进行重新构造形状
    label = mx.symbol.Reshape(data=label, target_shape=(0,))

    # 5. 网络模型输出
    return mx.symbol.SoftmaxOutput(data=fc2, label=label, name="softmax")


def Accuracy(label, pred):
    """
    准确率计算函数
    """
    # TODO: 如果不知道这里的label和pred的类型，可以进行输出
    # print(type(label))
    # print(type(pred))
    label = label.T.reshape((-1,))
    hit = 0
    total = 0
    for i in range((int)(pred.shape[0] / 7)):
        ok = True
        for j in range(7):
            k = i * 7 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total


def train():
    """模型训练方法，进行模型训练，并将训练模型结果输出"""
    # 1. 获取网络对象(神经网络对象)
    network = get_ocrnet()

    # 2. 模型构建(神经网络中的前馈操作)
    model = mx.model.FeedForward(
        symbol=network,  # 网络对象
        num_epoch=1,  # 模型训练的次数
        learning_rate=0.001,  # 学习率
        wd=0.00001,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),  # 给定初始参数
        momentum=0.9)

    # 3. 获取训练数据和测试数据集
    batch_size = 20
    data_train = OCRIter(20000000, batch_size, 7, 30, 120)
    data_test = OCRIter(1000, batch_size, 7, 30, 120)

    # 4. 设置log日志
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    # 5. 开始进行模型训练
    model.fit(X=data_train, eval_data=data_test, eval_metric=Accuracy,
              batch_end_callback=mx.callback.Speedometer(batch_size, 50))

    # 5. 模型保存
    # TODO: 这个模型输出有一个bug，在输出形成的json文件中，"Reshape"操作中，是没有参数: shape的，如果输出形成的文件中存在这个参数，那么需要将该参数删除
    print("模型训练完成，开始保存.....")
    model.save("cnn-ocr")


if __name__ == '__main__':
    train()
