import random

from numpy import exp, array, random, dot


class NeuralNetWork(object):
    def __init__(self):
        # 指定随机数种子，保证每次出现的是同一个随机数
        random.seed(1)
        self.dendritic_weight = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.dendritic_weight))

    def train(self, training_inputs, training_outputs, number_of_training_iterators):
        for iteration in range(number_of_training_iterators):
            # 每次迭代都计算一次结果
            output = self.think(training_inputs)
            # 计算计算结果与真实结果的误差
            error = training_outputs - output
            # 处理误差，获得调整参数
            adjustment = dot(training_inputs.T, error * self.__sigmoid_derivative(output))
            # 根据调整参数计算新的权重
            self.dendritic_weight += adjustment


if __name__ == '__main__':
    nn = NeuralNetWork()
    print("初始树突权重：{}".format(nn.dendritic_weight))

    training_inputs_sample = array([[0, 0, 1],
                                    [1, 1, 1],
                                    [1, 0, 1],
                                    [0, 1, 1]])
    training_outputs_sample = array([[0, 1, 1, 0]]).T

    nn.train(training_inputs_sample, training_outputs_sample, 10000)

    print("训练后树突权重：{}".format(nn.dendritic_weight))

    i1 = int(input("输入第一个数："))
    i2 = int(input("输入第二个数："))
    i3 = int(input("输入第三个数："))

    result = nn.think(array([i1,i2,i3]))
    print("预测结果为：{}".format(result))
