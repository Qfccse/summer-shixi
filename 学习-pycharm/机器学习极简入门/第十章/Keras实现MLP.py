# 步骤
""""
1.导入数据 数据规模为 60000*（28,28）
2.将数据转化为一维向量 60000* 784
3.将向量归一
4.导入MLP模型
5.设置模型的输入神经元为784，隐藏神经元784
6.训练
7.输出训练结果，保存训练模型
"""

import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import PIL
from PIL import Image

#  预先下载mnist.npz，放在\.keras\datasets下
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
# 导入MNIST数据集

# 将原本28*28 images 转换 a 784 vector
num_pixels = train_images.shape[1] * train_images.shape[2]
x_train = train_images.reshape((train_images.shape[0], num_pixels)).astype('float32')
# 将数字图像images的数字标准化，即normalize input from 0-255 to 0-1
train_images_normalize = x_train / 255
x_test = test_images.reshape((test_images.shape[0], num_pixels)).astype('float32')
test_images_normalize = x_test / 255
# 将训练数据和测试数据的类别进行one-hot独热编码
from keras.utils import np_utils

train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(units=784, activation='relu', input_dim=784, kernel_initializer='normal'))
network.add(layers.Dense(units=10, kernel_initializer='normal', activation='softmax'))
print(network.summary())
# 建立多层感知器模型

network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# 编译网络

network.fit(train_images_normalize, train_labels, epochs=5, batch_size=200,
            validation_split=0.2, verbose=2)
# 训练模型

test_loss, test_acc = network.evaluate(test_images_normalize, test_labels)
# 评价模型

print('test_acc:', test_acc)
# 输出精度

filename = 'keras_mnistmodel.h5'
network.save(filename)
# 保存模型
