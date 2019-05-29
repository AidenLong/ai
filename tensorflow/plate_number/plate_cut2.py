# -- encoding:utf-8 --
"""
Create by ibf on 2018/5/28
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def show_image_tensor(image_tensor):
    # 要求：使用交互式会话
    # 获取图像tensor对象对应的image对象，image对象时一个[h,w,c]
    # print(image_tensor)
    image = image_tensor.eval()
    # print(image)
    print("图像大小为:{}".format(image.shape))
    if len(image.shape) == 3 and image.shape[2] == 1:
        # 黑白图像
        plt.imshow(image[:, :, 0], cmap='Greys_r')
        plt.show()
    elif len(image.shape) == 3:
        # 彩色图像
        plt.imshow(image)
        plt.show()


sess = tf.InteractiveSession()
path = '0.jpg'

# 读取图像数据并转换为[height, weight, channels]的格式
img = tf.image.decode_jpeg(tf.read_file(path), 3)
# 图像重置大小
img = tf.image.resize_images(img, size=(300, 300), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 将图像转换为黑白的图像
img = tf.image.rgb_to_grayscale(img)
old_img = img

# 边缘/轮廓信息提取
img = tf.cast(img, dtype=tf.float32)
img = tf.expand_dims(img, 0)
for i in range(1):
    sobel_gx = [-1, 0, +1, 0, 0, 0, -1, 0, +1]
    conv2d_filter = tf.constant(value=sobel_gx, dtype=tf.float32, shape=[3, 3, 1, 1])
    img = tf.nn.conv2d(img, filter=conv2d_filter, strides=[1, 1, 1, 1], padding='SAME')
img = img[0]

# 二值化，小于等于170的设置为0，大于170设置为255
img = tf.where(tf.less_equal(img, 170), tf.zeros_like(img), tf.ones_like(img) * 255)

# 将白色的区域(车牌区域就是白色的)，扩大/膨胀 ==> 将这些白色区域的周边也变成白色的 ==> 只需要做一个maxpool
img = tf.expand_dims(img, 0)
# 膨胀
for i in range(2):
    img = tf.nn.max_pool(img, ksize=(1, 2, 3, 1), strides=(1, 1, 1, 1), padding='SAME')

# 将有一些膨胀不太好的地方，还原成黑色
img = tf.nn.max_pool(img * -1, ksize=(1, 3, 1, 1), strides=(1, 1, 1, 1), padding='SAME')
img = img * -1

# 继续膨胀一次
for i in range(5):
    img = tf.nn.max_pool(img, ksize=(1, 1, 5, 1), strides=(1, 1, 1, 1), padding='SAME')
img = img[0]

print(img.get_shape())
show_image_tensor(img)
