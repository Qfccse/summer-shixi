import numpy as np
import pandas as pd

a = np.random.randint(3,5,(2,2))

s1 = pd.Series(range(12),index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd'],
                                [0,1,2,0,1,2,0,1,2,0,1,2]])

print(s1)

print(s1.index)
# 获取
print(s1['c'])

# 同样也可以用高级索引
print(s1[:,2])

#交换两个索引的层级
print(s1.swaplevel())

#排序，去重
print(s1.swaplevel().sort_index())