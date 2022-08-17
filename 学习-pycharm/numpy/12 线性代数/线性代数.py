import numpy.matlib
import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[11, 12],
              [13, 14]])
print(np.dot(a, b))
print(np.matmul(a, b))
print(np.inner(a, b))
print(np.vdot(a, b))

aa = np.array([1, 2])
bb = np.array([3, 4])
print(np.dot(aa, bb))
print(np.matmul(aa, bb))
print(np.inner(aa, bb))
print(np.vdot(aa, bb))
