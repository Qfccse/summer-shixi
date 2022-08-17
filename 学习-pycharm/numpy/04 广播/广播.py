import numpy as np

arr = np.array([10, 20, 30])
a = np.array([[0, 0, 0],
             [10, 10, 10],
             [20, 20, 20],
             [30, 30, 30]])
b = np.array([1, 2, 3])
print(arr * b)
print(a + b)

# tile(arr, (n,m)) 将 arr 在列上复制n-1次，行上复制m-1次
bb = np.tile(b, (4,1))
print(bb)