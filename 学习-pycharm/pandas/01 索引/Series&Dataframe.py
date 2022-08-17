import numpy as np
import pandas as pd

s = pd.Series([1, 2, 5, np.nan, 6, 8], pd.Index(['a', 'b', 'c', 'd', 'e', 'f']))
sb = s > 2
print(s)
print(s[0:2])
print(sb)
print(s[sb])

a = pd.DataFrame(np.arange(9).reshape(3, 3), index=['+', '-', '*'], columns=['a', 'b', 'c'])
print(a)
print(a[['a', 'c']])

print("基于索引名的高级索引 loc")
# series 用loc与不用没有区别
print(a.loc['+':'*', 'a'])

print("基于索引序号的高级索引 iloc")
print(a.iloc[0:3, 0])

# print("混合高级索引 ix")  已弃用
# print(a.ix[0:3, 'a'])