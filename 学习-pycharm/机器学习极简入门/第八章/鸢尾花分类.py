import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

datas = load_iris()
iris_x = datas.data
iris_y = datas.target
df = pd.DataFrame(iris_x, columns=datas.feature_names)
print(df[:5])
print(iris_y[:2])
print('数据集规格大小', iris_x.shape)
print('标签有', datas.target＿names)
print("数据属性有", datas.feature_names)

x_train, x_test,  y_train,y_test = train_test_split(iris_x, iris_y, test_size=0.3, random_state=42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
pridict_y = nb.predict(x_test)
print('真实值', y_test)
print('预测值', pridict_y)
print('nb_train_score' , nb.score(x_train, y_train))
print('nb_test_score', nb.score(x_test,y_test))

from sklearn.naive_bayes import BernoulliNB

nbb = BernoulliNB()
nbb.fit(x_train, y_train)
pridict_y = nbb.predict(x_test)
print('真实值', y_test)
print('预测值', pridict_y)
print('bb_train_score' , nbb.score(x_train, y_train))
print('bb_test_score', nbb.score(x_test,y_test))