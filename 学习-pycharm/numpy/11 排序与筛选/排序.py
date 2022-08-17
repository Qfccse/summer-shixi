import numpy as np

# sort(a, axis=-1, kind=None, order=None) 排序，不改变arr
# argsort(...) 返回排序后的数组在arr中的索引序列
# 为了降序排序，可以先升序排序然后在反转数组 arr = arr[::-1]
# sort_complex 复数排序

nm = ('raju', 'anil', 'ravi', 'amar')
dv = ('f.y.', 's.y.', 's.y.', 'f.y.')
# 先按 nm 如果nm相同则按照dv
ind = np.lexsort((dv, nm))
print('调用 lexsort() 函数：')
print(ind)
print('\n')
print('使用这个索引来获取排序后的数据：')
print([nm[i] + ", " + dv[i] for i in ind])