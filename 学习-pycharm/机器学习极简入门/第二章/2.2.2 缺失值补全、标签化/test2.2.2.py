import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

imp = SimpleImputer(missing_values=np.NaN, strategy="mean")
print("##########缺失值补全##########")
imp.fit([[1, 2],
         [np.nan, 3],
         [7, 6]])
x = [[np.nan, 2],
     [6, np.nan],
     [7, 6]]
print(x)
# 处理需要补全的数据
# 用 fit() 的矩阵的列均值来补全 x的列中的缺失值
print(imp.transform(x))
print("###LabelEncoder_标准化标签，将标签值统一转换成range（标签值个数-1）范围内#")

data = ["Japan", "China", "Japan", "Korea", "China"]
print(data)
le = preprocessing.LabelEncoder()
le.fit(data)
# 将data中的元素编码
print("标签个数 %s" % le.classes_)
print("标签值标准化 %s " % le.transform(data))
data2 = ["Japan", "China", "China", "Korea", "Korea"]
print(data2)
print("标签值标准化 %s " % le.transform(data2))
