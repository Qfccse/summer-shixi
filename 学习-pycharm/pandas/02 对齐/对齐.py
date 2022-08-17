import numpy as np
import  pandas as pd

a = pd.Series(range(10,20),index=range(10))
b = pd.Series(range(20,25),index=range(5))
print(a)
print(b)
print(a+b)

ad = pd.DataFrame(np.arange(16).reshape(4,4),index=range(4),columns=range(4))
bd = pd.DataFrame(np.arange(9).reshape(3,3),index=range(3),columns=range(3))
print(ad)
print(bd)
print(ad+bd)

c = a.add(b,fill_value=int(0))
print(a)
print(c)

print(b.add(a,fill_value=200))

s1 = pd.DataFrame(np.arange(4).reshape(2,2))
s2 = pd.DataFrame(np.arange(4).reshape(2,2))

print(s1.mul(s2))