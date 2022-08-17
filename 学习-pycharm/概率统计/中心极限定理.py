import numpy as np
import matplotlib.pyplot as plt
#假设观测某人抛掷质地均匀的骰子，也就是说掷出1～6的概率都是1/6
random_data = np.random.randint(1, 7, 60000)
#展示抛掷骰子的数值情况
print(random_data)
# 展示抛掷骰子结果的均值和标准差
print ('均值为： ',random_data.mean())
print ('标准差为：',random_data.std())
samples_100_mean = []

for i in range(0,10000):
    sample = []
    for j in range(0,100):
        sample.append(random_data[int(np.random.random()*len(random_data))])
    samples_100_mean.append((np.mean(sample)))
    samples_many_means = np.mean(samples_100_mean)
print("多次抽取样本的总均值", samples_many_means)
plt.hist(samples_100_mean,bins=200,density=0)


#直方图展示抛掷骰子的结果
#plt.hist(random_data,bins=50,density=0)
plt.show()