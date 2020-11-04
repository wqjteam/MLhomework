import matplotlib.pyplot as plt
import numpy as np
import time
from math import *

plt.ion() #开启interactive mode 成功的关键函数
plt.figure(1)
t = [0]
t_now = 0
m = [sin(t_now)]

for i in range(2000):
    # plt.clf() # 清空画布上的所有内容。此处不能调用此函数，不然之前画出的点，将会被清空。
    t_now = i*0.1
    """
    由于第次只画一个点，所以此处有两种方式，第一种plot函数中的样式选
    为点'.'、'o'、'*'都可以，就是不能为线段'-'。因为一条线段需要两
    个点才能确定。第二种方法是scatter函数，也即画点。
    """
    plt.plot(t_now,sin(t_now),c='r') # 第次对画布添加一个点，覆盖式的。
    # plt.scatter(t_now, sin(t_now))

    plt.draw()#注意此函数需要调用
    plt.pause(0.01)