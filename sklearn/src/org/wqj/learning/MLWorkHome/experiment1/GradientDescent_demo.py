# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

# 目标函数:y=x^2
def func(x):
    return np.square(x)
#exp指数
#sqrt 平方根
#square 平方


# 目标函数一阶导数也即是偏导数:dy/dx=2*x
def dfunc(x):
    return 2 * x


def GD(x_start, df, epochs, lr):
    """
    梯度下降法。给定起始点与目标函数的一阶导函数，求在epochs次迭代中x的更新值
    :param x_start: x的起始点
    :param df: 目标函数的一阶导函数
    :param epochs: 迭代周期
    :param lr: 学习率
    :return: x在每次迭代后的位置（包括起始点），长度为epochs+1
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x) #传递函数 进行平方
        # v表示x要改变的幅度
        v = - dx * lr
        x += v
        xs[i+1] = x
    return xs


def demo_GD():
    # 演示如何使用梯度下降法GD()
    line_x = np.linspace(-5, 5, 100) #进行均分
    line_y = func(line_x)

    x_start = -5
    epochs = 10000

    lr = 0.01
    x = GD(x_start, dfunc, epochs, lr=lr)

    color = 'r'
    plt.plot(line_x, line_y, c='b')
    plt.plot(x, func(x), c=color, label='lr={}'.format(lr))
    plt.scatter(x, func(x), c=color, )
    plt.legend()
    plt.show()
demo_GD()