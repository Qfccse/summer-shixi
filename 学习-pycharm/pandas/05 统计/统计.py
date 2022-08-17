# sum, mean, max, min
import numpy as np
import pandas as pd

arr1 = np.random.rand(4,3)
pd1 = pd.DataFrame(arr1,columns=list('ABC'),index=list('abcd'))
f = lambda x: '%.2f'% x
pd2 = pd1.applymap(f).astype(float)
print(pd2)
#默认把这一列的Series计算,所有行求和
print(pd2.sum())
#指定求每一行的所有列的和
print(pd2.sum(axis='columns'))
#查看每一列所有行的最大值所在的标签索引，同样我们也可以通过axis='columns'求每一行所有列的最大值的标签索引
print(pd2.idxmax())

print(pd2.A.describe())

