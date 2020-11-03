# coding=utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("sklearn\\") + len("sklearn\\")]
dataPath = rootPath + "Input/MLWorkHome/experiment1/data1.csv"
dataSet = pd.read_csv(dataPath, header=None)
price = []
rooms = []
area = []
for data in range(0, len(dataSet)):
    area.append(dataSet[0][data])
    rooms.append(dataSet[1][data])
    price.append(dataSet[2][data])


# 根据theta也就是权值计算预估值
def return_y_estimate(theta, data_x):
    return np.dot(data_x, theta)


# 计算loss
def return_j(theta, data_x, data_y):
    N = data_x.shape[0]  # 一共多少组数据
    estimate_y = return_y_estimate(theta, data_x)
    temp = data_y - estimate_y  # 相当于 ytrue-y预估
    loss = np.dot(temp.T, temp) / 2 * N
    return loss


# 对loss求偏导,也就是gradient
def return_dj(theta, data_x, data_y):
    N = data_x.shape[0]
    estimate_y = return_y_estimate(theta, data_x)
    number_theta = data_x.shape[1]  # 一共有number_theta个theta
    DJ = np.zeros([number_theta, 1])
    # 每个theta都需要去求偏导,让在每个theta方向的损失都降低到最小(单个叫求导,多个叫梯度)
    for i in range(number_theta):
        # 对xi求偏导,data_x[:,i]表示取第一列,第二列,第三列
        estimate_y[i, 0] = np.dot((estimate_y - data_y).T, data_x[:, i]) / N
    return DJ


def gradient_descent(data_x, data_y, Learning_rate=0.01, ER=0.01, MAX_LOOP=10000):
    # 把X多家一列,看走b,亦为X0
    number_datax = data_x.shape[0]  # 获取多少行数据
    new_column = np.ones([number_datax, 1])  # 生成一列全为1的列值
    new_data_x = np.column_stack((data_x, new_column))
    theta = np.mat(np.random.normal(0, 1, [number_datax + 1, 1]))
    loss = np.zeros([1, MAX_LOOP]).flatten()
    print(loss)
    for i in range(MAX_LOOP):
        last_theta = theta  # 保存上次的theta的值
        gradient = return_dj(theta, new_data_x, data_y)
        theta = theta - Learning_rate * gradient  # 为梯度优化公式
        # 判断准确率是否小于某个认为的值,如果小于的话,直接跳出
        loss[i] = return_j(theta, new_data_x, data_y)
        er = abs(return_j(last_theta, new_data_x, data_y) - return_j(theta, new_data_x,
                                                                     data_y))  # 对比上次和这次的采用不同的theta的区别,计算loss abs绝对值
        if er < ER:
            return loss, theta
    return loss, theta


if __name__ == '__main__':
    # 演示如何使用梯度下降法GD()
    line_x = np.linspace(0, 10000, 10000)  # 进行均分
    data_y = np.mat(price).reshape(-1,1)
    data_x = np.column_stack((np.mat(rooms).reshape(-1,1), np.mat(area).reshape(-1,1)))
    loss, theta = gradient_descent(data_x, data_y)
    plt.plot(line_x, loss, c='r')
    plt.legend()
    plt.show()
