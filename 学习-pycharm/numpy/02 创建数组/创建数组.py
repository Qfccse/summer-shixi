import numpy as np
import numpy.matlib

print("根据已有数组创建")
a = np.array([[1, 2, 3], [2, 2, 9]])
print(a)

arr = [[1, 2, 3], [4, 5, 6]]
a2 = np.asarray(arr)
# 当数据源是ndarray时，array会copy出一个副本，占用新的内存，但asarray不会,与ndarray共享内存。
# 当数据源是元组时，没有区别
# a2 = np.array(arr, dtype=dt)

str = b"hello "
a22 = np.frombuffer(str, dtype='S2')
print(a22)

llist = range(10)
print(llist)
a23 = np.fromiter(llist,dtype='i4')
print(a23)

print("*"*50)
print("凭空创建")
# 空数组，随机元素
a3 = np.empty((3, 2))
print(a3)


a32 = np.matlib.rand((3,2))

a4 = np.zeros((3, 2))
print(a4)

a5 = np.ones((3, 2))
print(a5)

# 单位矩阵
a6 = np.identity(5)
print(a6)


# 只包含左端
a7 = np.arange(1, 10, 1)
print(a7)

# 包含两端
a8 = np.linspace(1, 9, 10)
print(a8)
