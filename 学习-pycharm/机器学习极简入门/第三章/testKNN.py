import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

feat_cols = ['mass', 'width', 'height', 'color_score']

data = pd.read_csv('fruit_data.csv')

fruit2num = {'apple': 0,
             'mandarin': 1,
             'orange': 2,
             'lemon': 3}
data['fruit_label'] = data['fruit_name'].map(fruit2num)

x = data[feat_cols].values
y = data['fruit_label'].values

x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(x, y, random_state=20, test_size=0.2)
print('原始数据集共{}个样本，其中训练样本有{}，测试样本有{}'.format(x.shape[0], x_train_set.shape[0], x_test_set.shape[0]))

# 模型的n_neighbors 代表邻近数， weight 表示以什么为权重，p2 代表采用欧式距离
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)
# 通过fit来训练模型
knn_model.fit(x_train_set, y_train_set)
# 准确率检测
accur = knn_model.score(x_test_set, y_test_set)

print("准确率为", accur)

num2fruit = dict(zip(fruit2num.values(), fruit2num.keys()))

for idx in range(y_test_set.shape[0]):
    test_feat = [x_test_set[idx]]
    y_predict = num2fruit.get(int(knn_model.predict(test_feat)))
    y_real = num2fruit.get(y_test_set[idx])
    YorN = 'yes' if y_predict == y_real else 'no'
    print(f'第{idx + 1}个测试水过的结果是{y_predict}，' f'本该是{y_real},所以测试{YorN}')
