# pip install scikit-learn --upgrade
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 载入数据
mnist = fetch_openml('mnist_778', data_home='./scikit_learn_data')
print('样本数量：{},样本特征数：{}'.format(mnist.data.shape[0], mnist.data.shape[1]))
