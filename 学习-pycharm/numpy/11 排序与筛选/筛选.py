import numpy as np
a = np.array([3, 4, 2, 1])
# 相当于快排里的
print(np.partition(a, 3))
# 将数组 a 中所有元素（包括重复元素）从小到大排列，3 表示的是排序数组索引为 3 的数字，比该数字小的排在该数字前面，比该数字大的排在该数字的后面
print(np.partition(a, (1, 3)))
# 小于 1 的在前面，大于 3 的在后面，1和3之间的在中间


print(a[np.argpartition(a, 2)[2]])

a = np.array([[30, 40, 0], [0, 20, 10], [50, 0, 60]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 nonzero() 函数：,返回的是行索引和列索引分开')
print(np.nonzero(a))

x = np.arange(9.).reshape(3, 3)
print('我们的数组是：')
print(x)
# 定义条件, 选择偶数元素
condition = np.mod(x, 2) == 0
print('按元素的条件值：')
print(condition)
print('使用条件提取元素：')
print(np.extract(condition, x))
x = np.arange(9.).reshape(3,  3)
print ('我们的数组是：')
print (x)
print ( '大于 3 的元素的索引：')
y = np.where(x >  3)
print (y)
print ('使用这些索引来获取满足条件的元素：')
print (x[y])