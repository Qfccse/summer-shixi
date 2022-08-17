import numpy as np
import numpy as py

a = np.array([1,2,3],[4,5,6])

ac = a.copy()
af = a.flatten()
ar = a.ravel()


print(ac is af)
print(ac is ar)
print(af is ar)
# 但因为ar是a的一种展示方式,虽然他们是不同的对象,但在修改ar的时候,a中相应的数也改变了
ar[1] = 999
print(a)



