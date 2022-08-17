import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 训练数据   根据身高  体重  鞋号预测男女性别
X = np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
              [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
              [159, 55, 37], [171, 75, 42], [181, 85, 43]])

y = ['male', 'male', 'female', 'female',
     'male', 'male', 'female', 'female',
     'female', 'male', 'male']

lec = LabelEncoder()
y = lec.fit_transform(y)
print(y)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(X)

knn = KNeighborsClassifier()
knn.fit(X, y)

pre = knn.predict(np.array([[167, 61, 43],
                            [178, 77, 44],
                            [155, 60, 39],
                            [178, 71, 44]]))
print(pre)
