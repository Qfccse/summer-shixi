
index = [2837,  2947,  2972,  3230,  3321,  3448,  3650,  5448,  5491,
         6123,  7580,  7670,  8318,  8559,  9380, 10080, 12757, 12792,
         13184, 16260]

import pandas as pd
import numpy as np
df = pd.read_csv(r"red_eye.csv", index_col=0, header=None)
a = np.array(df)
print(a)
b = []
for i in range(len(index)):
    b.append(a[index[i]-1])

b = pd.DataFrame(b)

b.to_csv('20_f.csv')