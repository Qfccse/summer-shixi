import numpy as np

a, b = 13, 17
# 输出二进制
print(bin(a))
# 按位取反
print(bin(np.invert(a)))
# 转为无符号数
print(bin(np.uint32(b)))
print(bin(np.invert(np.uint32(b))))

# 按位与
print(np.bitwise_and(a, b))
# 按位或
print(np.bitwise_or(a, b))

# left_shift， right_shift 左移和右移
