from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np
import pandas as pd

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)

import matplotlib.pyplot as plt
# plt.imshow(train_images[0])
# plt.show()

train_images_r = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images_r = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

train_images_r = train_images_r / 255
test_images_r = test_images_r / 255

train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

# 创建CNN网络
def create_model():
    model = Sequential()
    # padding 使用 same 和 valid
    # same 的准确率稍高
    # valid 的迭代速度更快
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu', padding='valid'))
    # padding=same 时，在池化过程中，如果剩余的宽度不足2，则在外围补0
    # padding=valid时，如果剩余宽度不足2，则抛弃
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, (3, 3),activation='relu', padding='valid'))
    # 多加一层卷积核池化在 epoch = 10 时训练时间增加了1/4，但是准确率却有下降
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


cnn_model = create_model()

cnn_model.fit(train_images_r, train_labels, validation_data=(test_images_r, test_labels),
              epochs=20, batch_size=200, verbose=2)

score = cnn_model.evaluate(test_images_r, test_labels)
print(score[1])
