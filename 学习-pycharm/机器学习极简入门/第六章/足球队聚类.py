import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

data = pd.read_csv("footballdata.csv", encoding='utf-8')
train_x = data[['2019年国际排名', '2018年世界杯', '2015年亚洲杯']]

scaler = preprocessing.MinMaxScaler()
train_x = scaler.fit_transform(train_x)

kmean = KMeans(n_clusters=3)
kmean.fit(train_x)
predict_y = kmean.predict(train_x)

result = pd.concat((data, pd.DataFrame(predict_y)),axis=1)
result.rename({0:'聚类'}, axis=1, inplace=True)

print(result)

