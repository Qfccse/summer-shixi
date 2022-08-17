import numpy as np
import  pandas as pd

s4 = pd.Series(range(10, 15), index = np.random.randint(5, size=5))
print(s4)

# 给索引排序
# 排序默认使用升序排序，ascending=False 为降序排序
s4_sort = s4.sort_index() # 0 0 1 3 3
print(s4_sort)

# DataFrame
df4 = pd.DataFrame(np.random.randn(3, 5),
                   index=np.random.randint(3, size=3),
                   columns=np.random.randint(5, size=5))
print(df4)

# 先按照列索引降序排序，在按照行索引升序排序
df4_isort = df4.sort_index(axis=1, ascending=False).sort_index()
print(df4_isort) # 4 2 1 1 0
