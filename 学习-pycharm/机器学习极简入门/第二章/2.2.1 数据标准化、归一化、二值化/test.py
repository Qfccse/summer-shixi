import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer

data = np.array([[3, -1.7, 3.5, -6],
                 [0, 4, -0.3, 2.5],
                 [1, 3.5, -1.8, -4.5]])
data_mean = np.mean(data)
data_std = np.std(data)
min_data = np.min(data, axis=0)
max_data = np.max(data, axis=0)
print("原始数据：")
print(data)
print("均值：")
print(data_mean)
print("方差：")
print(data_std)
print("min")
print(min_data)
print("max")
print(max_data)

# X' = (X- 均值) / 标准差
data_standardscaler = StandardScaler().fit_transform(data)
print("原始数据使用StandardScaler进行数据标准化处理后：")
print(data_standardscaler)

# X' = (X-Xmin) / (Xmax-Xmin)
data_minmaxscaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
print("原始数据使用MinMaxScaler进行数据归一化处理后：")
print(data_minmaxscaler)

# X' = x>0?1:0
data_binarizer = Binarizer().fit_transform(data)
print("原始数据使用Binarizer进行数据二值化处理后：")
print(data_binarizer)
