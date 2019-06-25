# -*- coding:utf-8 -*-
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, LeakyReLU
from keras.layers import Dense, Dropout, BatchNormalization, Activation, UpSampling2D, Reshape, Conv2DTranspose
from skimage.io import imsave

batch_size = 16
epochs = 1000
image_height = 32
image_weight = 32
channel = 3
image_size = image_height * image_weight
z_size = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('x_train.shape = ' + str(x_train.shape))
print('y_train.shape = ' + str(x_test.shape))
print('train samples = ' + str(x_train.shape[0]))
print('test samples = ' + str(x_test.shape[0]))


# 构建生成模型
def build_generator(z_prior):
    G = Sequential()
    depth = 256
    # dropout = 0.4
    dim = 8
    G.add(Dense(dim * dim * depth, input_dim=z_prior))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Reshape((dim, dim, depth)))
    G.add(UpSampling2D())
    G.add(Conv2DTranspose(depth, 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(int(depth / 16), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(3, 5, padding='same'))
    G.add(Activation('sigmoid'))
    G.summary()
    return G


# 构建判别模型
def build_disciminator():
    D = Sequential()
    depth = 128
    dropout = 0.4
    input_shape = (image_height, image_weight, channel)
    D.add(Conv2D(depth, 2, strides=2, padding='same', input_shape=input_shape))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth * 2, 2, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth * 4, 2, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth * 8, 2, strides=1, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Flatten())
    D.add(Dense(1, activation='sigmoid'))
    D.summary()
    return D


# 训练判别模型
def build_DM(D):
    DM = Sequential()
    DM.add(D)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0002, decay=6e-8)
    DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return DM


# 训练对立模型
def build_AM(D, G):
    AM = Sequential()
    AM.add(G)
    AM.add(D)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=3e-8)
    AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return AM


# 图片保存
def show_result(x_gen_val, fname):
    _, img_h, img_w, _ = x_gen_val.shape
    grid_h = img_h * 4 + 5 * (4 - 1)
    grid_w = img_w * 4 + 5 * (4 - 1)
    img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, res in enumerate(x_gen_val):
        if i >= 4 * 4:
            break
        img = res * 255
        img = img.astype(np.uint8)
        row = (i // 4) * (img_h + 5)
        col = (i % 4) * (img_w + 5)
        img_grid[row: row + img_h, col: col + img_w, :] = img
    imsave(fname, img_grid)


if __name__ == '__main__':
    D = build_disciminator()
    DM = build_DM(D)
    G = build_generator(z_size)
    AM = build_AM(D, G)
    noise = np.random.uniform(0.0, 1.0, size=(batch_size, 100))
    for i in range(epochs):
        # Train discriminator 获取真实图片
        real_img = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
        # 生成图片
        fake_img = G.predict(noise)
        show_result(real_img, 'output_sample/real_%d.jpg' % i)
        if i > 0 and i % 10 == 0:
            show_result(fake_img, 'output_sample/test_%d.jpg' % i)
        x = np.concatenate((real_img, fake_img))
        y = np.ones((batch_size * 2, 1))  # real label
        y[batch_size:, :] = 0  # fake label

        d_loss = DM.train_on_batch(x, y)
        # Train adversarial
        x = np.random.uniform(0.0, 1.0, size=(batch_size, 100))
        y = np.ones((batch_size, 1))  # target: all output 1

        a_loss = AM.train_on_batch(x, y)
        # Log
        print('(%d/%d) [D loss: %f, acc: %f] [A loss: %f, acc: %f]'
              % (i + 1, epochs, d_loss[0], d_loss[1], a_loss[0], a_loss[1]))
