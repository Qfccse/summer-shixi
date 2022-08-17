# from sklearn import datasets
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import numpy as np
#
# iris = datasets.load_iris()
# x = iris.data[:, 2:4]
#
# print(x.shape)
# plt.scatter(x[:, 0], x[:, 1], c='red', marker='o', label='see')
# plt.xlabel('length')
# plt.ylabel('height')
#
# plt.show()
#
# kmean = KMeans(n_clusters=3)
# kmean.fit(x)
# label_predict = kmean.labels_
#
# x0 = x[label_predict == 0]
# x1 = x[label_predict == 1]
# x2 = x[label_predict == 2]
# plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='o', label='label1')
# plt.scatter(x1[:, 0], x1[:, 1], c='blue', marker='*', label='label2')
# plt.scatter(x2[:, 0], x2[:, 1], c='green', marker='+', label='label3')
# plt.xlabel('length')
# plt.ylabel('height')
# plt.legend(loc=2)
# plt.show()

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = load_iris()
x = iris.data[:,2:4]
plt.scatter(x[:,0],x[:,1])
plt.show()

km = KMeans(n_clusters=3)
km.fit(x)
x0 = x[km.labels_==0]
x1 = x[km.labels_==1]
x2 = x[km.labels_==2]
plt.scatter(x0[:,0],x0[:,1],c='red')
plt.scatter(x1[:,0],x1[:,1],c='blue')
plt.scatter(x2[:,0],x2[:,1],c='black')
plt.show()