import math
from sklearn.preprocessing import MinMaxScaler
movie_data = [[45, 2, 9, "喜剧片"],
              [21, 17, 5, "喜剧片"],
              [54, 9, 11, "喜剧片"],
              [39, 0, 31, "喜剧片"],
              [5, 2, 57, "动作片"],
              [3, 2, 65, "动作片"],
              [2, 3, 55, "动作片"],
              [6, 4, 21, "动作片"],
              [7, 46, 4, "爱情片"],
              [9, 39, 8, "爱情片"],
              [9, 38, 2, "爱情片"],
              [8, 34, 17, "爱情片"]]
scaler = MinMaxScaler()
scaler.fit_transform(movie_data[:,:2])
KNN = []
x = [23, 3, 17]
for v in movie_data:
    d = math.sqrt((x[0] - v[0]) ** 2 + (x[1] - v[1]) ** 2 + (x[2] - v[2]) ** 2)
    KNN.append([round(d, 2), v[3]])

KNN.sort(key=lambda k: k[0])

feature = KNN[:5]

label = {'爱情片': 0, "喜剧片": 0, "动作片": 0}

print(feature)

for f in feature:
    label[f[1]] += 1

print(label)
print(label.items())
label = sorted(label.items(),key=lambda x:x[1],reverse=True)

print(label)