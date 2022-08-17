from sklearn import datasets
from sklearn.model_selection import train_test_split
boston = datasets.load_boston()
x = boston.data
y = boston.target

train_x,test_x,train_y,test_y = train_test_split(x,y,random_state=8)
print(train_x.shape)
print(test_x.shape)

from sklearn.svm import SVR

for kernel in {'linear','rbf'}:
    svr = SVR(kernel=kernel)
    svr.fit(train_x,train_y)
    print(kernel, '核函数得分：{:.3f}'.format(svr.score(test_x,test_y)))


from sklearn.preprocessing import StandardScaler
std = StandardScaler()

std.fit(train_x)

scaled_train_x = std.transform(train_x)
scaled_test_x = std.transform(test_x)

for kernel in {'linear','rbf'}:
    svr = SVR(kernel=kernel,C=100,gamma=0.1)
    svr.fit(scaled_train_x,train_y)
    print('在数据进行预处理后，并调整了C和gammer参数后')
    print(kernel, '核函数得分：{:.3f}'.format(svr.score(scaled_test_x,test_y)))













