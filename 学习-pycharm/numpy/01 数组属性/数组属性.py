import numpy as np

# i4 表示 int32 数据类型
# >i4 表示数据存放方式为大端存放
dt = np.dtype("i4")

arr = [[1, 2, 3],
              [4, 5, 3]]

a = np.asarray(arr, dtype=dt)
print(a)
print(a.shape)
print(a.ndim)   # 维度
print(a.dtype.name) # 数据类型
print(a.itemsize) # 数据字节
print(a.size)
print(a.flags)

# dt = np.dtype([('age',np.int32)])
# a = np.array([(10,),(20,),(30,)], dtype= dt)
# print(a)
# print(a.shape)
# print(a.ndim)   # 维度
# print(a.dtype.name) # 数据类型
# print(a.itemsize) # 数据字节
# print(a.size)
#
#
# # student 是一个结构体数据类型
# # 对于 a 中的('abc', 21, 50)， abc的变量明为name，类型为S20
# student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')])
# a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student)
# print(a)
