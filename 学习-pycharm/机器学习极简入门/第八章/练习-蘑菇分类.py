import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
mr_data = pd.read_csv('mushrooms.csv')
print(mr_data.head())

mr_data_f = mr_data.drop('class', axis=1)

le = LabelEncoder()
for i in mr_data_f.columns:
    mr_data_f[i] = le.fit_transform(mr_data_f[i])

f_train, f_test, p_train, p_test = train_test_split(mr_data_f, mr_data['class'], random_state=1)

bs = GaussianNB()
bs.fit(f_train, p_train)
pre_p = bs.predict(f_test)

print("伯努利贝叶斯待测模型评分：" + str(accuracy_score(p_test, pre_p)))

#
# import pandas as pd
# import numpy as np
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# mushrooms = pd.read_csv('mushrooms.csv')
# le = LabelEncoder()
# train = mushrooms.drop("class", axis=1)
# for i in train.columns:
#     train[i] = le.fit_transform(train[i])
#
# X_train, X_test, y_train, y_test = train_test_split(train, mushrooms["class"], random_state=1)
#
# clf = BernoulliNB(alpha=10)
# clf.fit(X_train, y_train)
# train_prediction = clf.predict(X_train)
# test_prediction = clf.predict(X_test)
# print("伯努利贝叶斯训练模型评分：" + str(accuracy_score(y_train, train_prediction)))
# print("伯努利贝叶斯待测模型评分：" + str(accuracy_score(y_test, test_prediction)))
