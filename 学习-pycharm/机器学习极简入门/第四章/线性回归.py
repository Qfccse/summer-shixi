from numpy import *
import numpy as np
m = 20

x0 = ones((m, 1))  # 生成全为 1 的列向量
x1 = arange(1, m + 1).reshape(m, 1)  # 生成 1 - m 的m列向量
x = hstack((x0, x1))
y = array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)

alpha = 0.01


def cost_function(theta, x, y):
    diff = dot(x, theta) - y
    return (1 / (2 * m)) * dot(diff.transpose(), diff)


def gradient_function(theta, x, y):
    diff = dot(x, theta) -y
    return (1 / m) * dot(x.T, diff)

# 梯度下降
def gradient_descent(x, y, alpha):
    theta = array([1, 1]).reshape(2, 1)
    grandient = gradient_function(theta, x, y)
    while not all(np.abs(grandient) <= 1e-5):
        theta = theta - alpha * grandient
        grandient = gradient_function(theta, x, y)
    return theta


optimal = gradient_descent(x, y, alpha)
print('optimal: ', optimal)
print('cost function: ', cost_function(optimal, x, y)[0][0])

def pplot(x,y,theta):
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    ax.scatter(x,y,s=30,c='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    x = arange(0,21,0.2)
    y = theta[0] + theta[1]*x
    ax.plot(x,y)
    plt.show()

pplot(x1,y,optimal)
