import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


crrp = pd.read_csv('数学建模/result.csv')
crrs = pd.read_csv('数学建模/result-spearman.csv')
crrp = np.array(crrp)
crrs = np.array(crrs)
crrw = (crrp)*0.4106 + (crrp)*(1-0.4106)

print(crrw.shape)
xx = crrw.shape[0]
crrw = np.sort(-crrw, axis=0)
y = -crrw.T
x = range(500)
plt.scatter(x, y[:, :500], s=1)
print(y[:,:30])
plt.show()



