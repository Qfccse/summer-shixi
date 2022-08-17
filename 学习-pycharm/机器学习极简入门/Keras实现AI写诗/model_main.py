"""
思路

1.读取诗歌数据集
2.统计数据集中有多少符合条件的诗，多少字符
3.统计字符出现频率
4.取出现频率在前三千对这些词进行热独编码
5.对取出的符合条件的诗进行采样，构建出训练集
6.构建rnn模型
7.分批次将训练集导入模型
8.设置为没次迭代完成后进行一次预测



"""

# 地址
poetry_data_path = "data/poetry.txt"
# 诗中出现以下字符则舍去
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
# 取3000个字作诗,其中包括空格字符
WORD_NUM = 3000
# 将出现少的字使用空格代替
UNKONW_CHAR = " "
# 根据前6个字预测下一个字，比如说根据“寒随穷律变，”预测下一句第一个字“春”
TRAIN_NUM = 6

# 保存诗词
poetrys = []
# 保存出现的字
all_word = []

with open(poetry_data_path, encoding='utf-8') as f:
    for line in f:
        # 依次获得诗的内容,去除里面的空格
        poetry = line.split(":")[1].replace(" ", "")
        flag = True
        # 如果在句子中出现  DISALLOWED_WORDS
        for dis_words in DISALLOWED_WORDS:
            if dis_words in poetry:
                flag = False
                break

        # 只需要五言诗
        if len(poetry) < 12 or poetry[5] != '，' or (len(poetry) - 1) % 6 != 0:
            flag = False

        if flag:
            for word in poetry:
                all_word.append(word)
            poetrys.append(poetry)

print("一共有：{}首诗，一共有{}个字符".format(len(poetrys), len(all_word)))

from collections import Counter

# 对字数进行统计，进行降序排序
counter = Counter(all_word)
word_count = sorted(counter.items(), key=lambda x: -x[1])
print(word_count[:20])
print(type(word_count))
# 这里的 * 是把list拆解层成一个个单个元素，zip 是重新将这些元素打包为元组
most_num_word, _ = zip(*word_count)

# 取前2999个字，然后在最后加" "
use_words = most_num_word[:WORD_NUM - 1] + (UNKONW_CHAR,)
print(use_words[-20:])

# word 到 id 的映射
word_id_dict = {word: index for index, word in enumerate(use_words)}
# id 到 word 的映射
id_word_dict = {index: word for index, word in enumerate(use_words)}
print(list(word_id_dict.items())[0:10])
print(list(id_word_dict.items())[0:10])

import numpy as np


def word_to_one_hot(word):
    """
    将一个字转化为热独编码形式
    :param word: 一个字
    :return: np.array
    """
    one_hot_word = np.zeros(WORD_NUM)
    # 生僻字变为空格
    if word not in word_id_dict.keys():
        word = UNKONW_CHAR
    index = word_id_dict[word]
    one_hot_word[index] = 1

    return one_hot_word


def pharase_to_one_hot(pharase):
    """
    将句子变为热独编码
    :param pharase: 一个句子
    :return: list
    """
    one_hot_pharase = []
    for pword in pharase:
        one_hot_pharase.append(word_to_one_hot(pword))

    return one_hot_pharase


print(word_to_one_hot('，'))
print(pharase_to_one_hot('，。'))

np.random.shuffle(poetrys)

x_train_word = []
y_train_word = []

for poetry in poetrys:
    for i in range(len(poetry)):
        # 根据 前 TRAIN_NUM个字预测最后一个
        x = poetry[i:i+TRAIN_NUM]
        y = poetry[i+TRAIN_NUM]
        if '\n' not in x and '\n' not in y:
            x_train_word.append(x)
            y_train_word.append(y)
        else:
            break

print(len(x_train_word))
print(x_train_word[:5])
print(y_train_word[:5])


# 搭建 rnn模型
from tensorflow import keras
from tensorflow.python.keras.models import Input, Model
from tensorflow.python.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.python.keras.callbacks import LambdaCallback, ModelCheckpoint

def build_model():
    print("build model")
    input_tensor = Input(shape=(TRAIN_NUM,WORD_NUM))
    rnn = SimpleRNN(512,return_sequences=True)(input_tensor)
    dropout = Dropout(0.6)(rnn)

    rnn = SimpleRNN(256)(dropout)
    dropout = Dropout(0.6)(rnn)
    dense = Dense(WORD_NUM,activation='softmax')(dropout)

    model = Model(inputs=input_tensor,outputs=dense)
    model.compile(loss='categorical_crossentropy',optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    return model


model = build_model()

import math
def get_batch(batch_size = 32):
    """
    源源不断的产生热独编码训练数据
    :param batch_size: 一次产生的训练数据的大小，defaults to 32
    :return:
    """
    #确定每轮有多少个批次batch
    # steps = math.ceil(len(x_train_word)/batch_size)
    steps = math.ceil(100000 / batch_size)
    while True:
        for i in range(steps):
            x_trian_batch = []
            y_train_batch = []
            x_trian_data = x_train_word[i*batch_size:(i+1)*batch_size]
            y_train_data = y_train_word[i*batch_size:(i+1)*batch_size]

            for x,y in zip(x_trian_data,y_train_data):
                x_trian_batch.append(pharase_to_one_hot(x))
                y_train_batch.append(word_to_one_hot(y))

            yield np.array(x_trian_batch),np.array(y_train_batch)

def predict_next(x):
    """
    根据x预测下一个字符
    :param x: 输入的数据
    :return: 最大的字符索引，有可能为 2999 也就是 “ ”
    """

    predict_y = model.predict(x)[0]
    index = np.argmax(predict_y)
    return index

def generate_sample_result(epoch, logs):
    """
    生成五言诗
    :param epoch: 训练的poches
    :param logs: 日志
    :return:
    """
    if epoch %1 ==0:
        predict_sen = '一朝春夏改，'
        predict_data = predict_sen
        # 生成的4句五言诗,包括表点符号共24个字符
        while len(predict_sen) < 24:
            x_data = np.array(pharase_to_one_hot(predict_data)).reshape(1,TRAIN_NUM,WORD_NUM)
            # 根据6个字符预测下一个字符
            y = predict_next(x_data)
            predict_sen = predict_sen + id_word_dict[y]
            # “寒随穷律变，” ——> “随穷律变，春”
            predict_data = predict_data[1:]+id_word_dict[y]
        # 将数据写入文件
        with open('out/out.txt', 'a',encoding='utf-8') as f:
            f.write(predict_sen+'\n')


batch_size = 2048
model.fit_generator(
    generator=get_batch(batch_size),
    verbose=True,
    # steps_per_epoch=math.ceil(len(x_train_word) / batch_size),
    steps_per_epoch=math.ceil(100000 / batch_size),
    epochs=30,
    callbacks=[
        ModelCheckpoint("poetry_model.hdf5", verbose=1, monitor='val_loss', period=1),
        # 每次完成一个epoch会调用generate_sample_result产生五言诗
        LambdaCallback(on_epoch_end=generate_sample_result)
    ]
)
