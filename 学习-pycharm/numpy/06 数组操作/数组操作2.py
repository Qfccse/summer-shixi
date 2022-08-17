import  numpy as np

a = np.arange(16).reshape(4,4)
# 按行优先风割
print(np.split(a,2))
print(np.vsplit(a,2))
# 按列优先分割
print(np.split(a,2,1))
print(np.hsplit(a,2))

b = np.arange(9).reshape(3,3)
print(b)
b = np.resize(b,(4,4))
print(b)

c = np.arange(4).reshape(2,2)
c2 = np.arange(4).reshape(2,2)
c3 = np.arange(4).reshape(2,2)
print(c)
# 变为一位数组后添加
print(np.append(c,[8,9]))
# 在行向量上添加
print(np.append(c2,[[8,9]],axis=0))
# 在列向量上添加
print(np.append(c3,[[8,9],[10,12]], axis=1))

# np.unique() 去重
