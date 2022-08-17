import numpy as np
import  pandas as pd

df = pd.DataFrame(np.random.randn(5,4) - 1)
print(df)

dd = pd.DataFrame(np.random.randn(5,4) - 1)
print(np.abs(df))
# 使用applymap应用全部数据
print(dd.applymap(lambda x :np.abs(x)))

# 使用apply应用行或列数据
#f = lambda x : x.max()
print(df.apply(lambda x : x.max()))
# 指定轴方向，axis=1，方向是行
print(df.apply(lambda x : x.max(), axis=1))


