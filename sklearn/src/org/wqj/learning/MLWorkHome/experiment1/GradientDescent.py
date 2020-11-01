# coding=utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt

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


# exp指数
# sqrt 平方根
# square 平方
# 原函数
# theta_now为参数(想要模拟的变量)
def return_Y_estimate(theta_now, data_x):
    # 确保theta_now为列向量
    theta_now = theta_now.reshape(-1, 1)
    # 输出为Y
    return np.dot(data_x, theta_now)


# 传入的data_x的最左侧列为全1，即设X_0 = 1
def return_dJ(theta_now, data_x, y_true):
    y_estimate = return_Y_estimate(theta_now, data_x)
    # 共有_N组数据
    _N = data_x.shape[0]
    # 求解的theta个数
    _num_of_features = data_x.shape[1]
    # 构建
    _dJ = np.zeros([_num_of_features, 1])

    for i in range(_num_of_features):
        _dJ[i, 0] = 2 * np.dot((y_estimate - y_true).T, data_x[:, i]) / _N

    return _dJ


# 计算J的值
# 传入的data_x的最左侧列为全1，即设X_0 = 1
def return_J(theta_now, data_x, y_true):
    # 共有N组数据
    N = data_x.shape[0]
    temp = y_true - np.dot(data_x, theta_now)
    # 组成平方  计算损失
    _J = np.dot(temp.T, temp) / N

    return _J


# 默认最大迭代次数为1e4
def gradient_descent(data_x, data_y, Learning_rate=0.01, ER=1e-10, MAX_LOOP=1e5):
    # 样本个数为
    _num_of_samples = data_x.shape[0]
    # 在data_x的最左侧拼接全1列
    X_0 = np.ones([_num_of_samples, 1])
    new_x = np.column_stack((X_0, data_x))
    # 确保data_y为列向量
    new_y = data_y.reshape(-1, 1)
    # 求解的未知元个数为
    _num_of_features = new_x.shape[1]
    # 初始化theta向量
    theta = np.zeros([_num_of_features, 1]) * Learning_rate
    flag = 0  # 定义跳出标志位
    last_J = 0  # 用来存放上一次的Lose Function的值
    ct = 0  # 用来计算迭代次数

    while flag == 0 and ct < MAX_LOOP:
        last_theta = theta
        # 更新theta
        gradient = return_dJ(theta, new_x, new_y)
        theta = theta - Learning_rate * gradient
        er = abs(return_J(last_theta, new_x, new_y) - return_J(theta, new_x, new_y))

        # 误差达到阀值则刷新跳出标志位
        if er < ER:
            flag = 1

        # 叠加迭代次数
        ct += 1

    return theta


if __name__ == '__main__':
    rooms_nadrr = np.mat(rooms).reshape(-1, 1)
    area_nadrr = np.mat(area).reshape(-1, 1)
    martix_x = np.column_stack((rooms_nadrr, area_nadrr))
    martix_y = np.mat(price).reshape(-1, 1)
    print(gradient_descent(martix_x, martix_y))
