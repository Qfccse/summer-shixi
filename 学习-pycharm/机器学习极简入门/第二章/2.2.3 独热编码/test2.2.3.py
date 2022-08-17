from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
data = [[0, 0, 3],
        [1, 1, 0],
        [0, 2, 1],
        [1, 0, 2]]
print("数据矩阵是4*3，即4个数据，3个维度")
print(data)
# 对于 0 - 2 列 状态码为 0:0->10 1->01
#                     1:0->100 1->010 2->001
#                     2:0->1000 1->0100 2->0010 3->0001
enc.fit(data)
x = [[0, 1, 3]]
print("要编码的参数")
print(x)
print("编码结果")
print(enc.transform(x).toarray()[0])
print(enc.transform(x).toarray()[0][0])


