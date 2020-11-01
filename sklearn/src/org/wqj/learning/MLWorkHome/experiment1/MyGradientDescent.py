# coding=utf-8
import numpy as np


# 根据theta也就是权值计算预估值
def return_y_estimate(theta, data_x):
    return np.dot(data_x, theta)


# 计算loss
def return_J(theta, data_x, data_y):
    N = data_x.shape[1]
    estimate_y = return_y_estimate(theta, data_x)
    temp = data_y - estimate_y  # 相当于 ytrue-y预估
    loss = np.dot(temp.T, temp) / 2*N
    return loss

def return_DJ():
    return