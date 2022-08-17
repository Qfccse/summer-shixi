import numpy as np
import pandas as pd

df4 = pd.DataFrame(np.random.randint(3, 12,(3,5)))
print(df4)
# 按值排序 by= 选定某一列
df4_vsort = df4.sort_values(by=0, ascending=False)
print(df4_vsort)
