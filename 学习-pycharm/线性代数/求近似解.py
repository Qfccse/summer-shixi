# test
import numpy as np
from scipy import linalg

a = np.array([[1, 1],
             [2, 1],
             [1, 1]])
b = np.array([0, 0, 2])

#计算 A^T * A
A_T_A = np.dot(a.T, a)
#计算A_T_A的逆
A_T_A_in = linalg.inv(A_T_A)
#计算
P = np.dot(np.dot(a,A_T_A_in), a.T)
x = np.dot(np.dot(A_T_A_in,a.T), b)
p = np.dot(P, b)

print("投影矩阵为")
print(P)
print("*"*50)
print("近似解为")
print(x)
print("*"*50)
print("投影向量为")
print(p)
print("*"*50)


