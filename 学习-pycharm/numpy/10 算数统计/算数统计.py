#  add()，subtract()，multiply() 和 divide()
# numpy.reciprocal() 函数返回参数逐元素的倒数
# numpy.power() 函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂
# numpy.mod() 计算输入数组中相应元素的相除后的余数。 函数 numpy.remainder() 也产生相同的结果

# 统计 https://www.runoob.com/numpy/numpy-statistical-functions.html

import numpy as np

a = np.arange(6).reshape(3, 2)
print('我们的数组是：')
print(a)
print('\n')
print('修改后的数组：')
wt = np.array([3, 5])
# 按 axis= 1，即行计算加权平均
print(np.average(a, axis=1, weights=wt))
print('\n')
print('修改后的数组：')
print(np.average(a, axis=1, weights=wt, returned=True))

print(np.amin(a))
print(np.amin(a,axis=0))

print(np.ptp(a,axis=1))
print('\n')
# 下面是对axis的理解
# 行 列 分别对应着 axis=1 axis=0
arr = np.arange(6).reshape(3, 2)
print(arr)
print(np.sum(arr))
print(np.sum(arr, axis=0))
print(np.sum(arr, axis=1))
