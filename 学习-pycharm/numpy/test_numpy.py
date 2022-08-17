# 自定义类型
import numpy as np

a = np.array([1, 2, 3])
b = np.array(a)
c = np.asarray(a)
a[1] = 100
print(b)
print(c)
