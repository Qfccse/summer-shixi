import numpy as np
import pandas as pd

s = pd.Series([1, 2, 5, np.nan, 6, 8],pd.Index(['a','b','c','d','e','f']))
print(s)

a = pd.DataFrame(np.arange(9).reshape(3,3),index=['+','-','*'],columns=['a','b','a'])
print(a.head())