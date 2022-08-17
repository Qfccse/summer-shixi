import numpy as np

arr = [1, 2, 3, 6, 4]
print(arr[::-1])

a = np.array([np.nan, 2, 3, 4, np.nan])
# 过滤掉nan
print(a[~np.isnan(a)])

a2 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
print(a2[[0, 1, 2], [0, 1, 0]])
print(a2[a2 > 5])
print(a2[1:3, [0,1,2]])