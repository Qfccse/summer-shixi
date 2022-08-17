# 导入红酒数据集
from sklearn.datasets import load_wine
# 导入MLP神经网络
from sklearn.neural_network import MLPClassifier
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split


wine = load_wine()
# 把数据拆分为训练集和数据集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=62)
print(x_train.shape, x_test.shape)
# 设定神经网络参数
# MLP的隐藏层数为 2 个， 每层有 100 个节点，最大迭代数为 1000
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=62)

# 拟合数据训练模型
mlp.fit(x_train, y_train)
# 输出模型得分
print("数据没有经过预处理模型得分：{:.2f}".format(mlp.score(x_test,y_test)))

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_pp = scaler.transform(x_train)
x_test_pp = scaler.transform(x_test)
mlp.fit(x_train_pp, y_train)
print("经过预处理后模型得分：{:.2f}".format(mlp.score(x_test_pp,y_test)))
MinMaxScaler(feature_range=(0, 1), copy=True)
MaxAbsScaler(copy=True)



