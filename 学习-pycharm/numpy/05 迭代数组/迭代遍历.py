import numpy as np

a = np.arange(6).reshape(2, 3)
# a.T a的转置
print(a)
for x in a:
    print(x)
print('\n以nditer迭代器遍历\n')
for x in np.nditer(a):
    print(x, end=',')
print('\n以flat迭代器遍历\n')
for x in a.flat:
    print(x, end=',')
print('\n以fortran的存储方式\n')
for x in np.nditer(a.copy(order='F')):
    print(x, end=',')

# flatten () 返回的是一个数组的的副本，新的对象；ravel () 返回的是一个数组的非副本视图
print('\n展开的数组\n')
print(a.flatten())

print('\n展开的数组\n')
print(a.ravel())

