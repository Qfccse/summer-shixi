import pandas as pd
import numpy as np

a= pd.DataFrame({'data1' : [1] * 4 + [2] * 4,
                 'data2' : np.random.randint(0, 4, 8)})

print(a)
print('\n')
# 可以指定对哪一列进行去重
print(a.duplicated())
print(a.drop_duplicates())

print(a.applymap(lambda x : x*x))

# 单个值替换单个值
print(a.replace(1, -100))

# 多个值替换一个值
print(a.replace([1, 3], -100))

# 多个值替换多个值
print(a.replace([1, 3], [-100, -200]))