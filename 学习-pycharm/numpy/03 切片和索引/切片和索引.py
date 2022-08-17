import numpy as np

print("一维数组")
a = np.arange(10)
print(a[2:7:2])
print(a[slice(2, 7, 2)])
print(a[2:])

print("二维数组")
a2 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
# 切片中 逗号，区分的是维度，冒号：区分的是索引，省略号… 用来代替全索引长度
# 当逗号数量大于 数组的维度-1 后，表示复合操作
print("第一行以后")
print(a2[1:])
print("第二行")
print(a2[1, ...])
print("第二列")
print(a2[..., 1])
print("第一列以后")
# 相当于 a2[:,1:]
# (a2[:,1:])
print(a2[..., 1:])
print("第二行 + ")
print(a2[1, ..., 1:])
