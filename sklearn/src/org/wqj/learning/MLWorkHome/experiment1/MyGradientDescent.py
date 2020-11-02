# coding=utf-8
import numpy as np


# 根据theta也就是权值计算预估值
def return_y_estimate(theta, data_x):
    return np.dot(data_x, theta)


# 计算loss
def return_J(theta, data_x, data_y):
    N = data_x.shape[0]     #一共多少组数据
    estimate_y = return_y_estimate(theta, data_x)
    temp = data_y - estimate_y  # 相当于 ytrue-y预估
    loss = np.dot(temp.T, temp) / 2*N
    return loss

def return_DJ(theta, data_x, data_y):
    N = data_x.shape[0]
    estimate_y=return_y_estimate(theta, data_x)
    number_theta = data_x.shape[1] #一共有number_theta个theta
    DJ=np.zeros([number_theta,1])
    # 每个theta都需要去求偏导,让在每个theta方向的损失都降低到最小(单个叫求导,多个叫梯度)
    for i in range(number_theta):
        # 对xi求偏导,
        estimate_y[i,0] = np.dot((estimate_y-data_y).T,data_x[:,i])/N
    return