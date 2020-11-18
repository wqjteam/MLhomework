# coding=utf-8
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("sklearn\\") + len("sklearn\\")]
dataPath = rootPath + "Input/MLWorkHome/experiment2/data.csv"
x_y_data = pd.read_csv(dataPath, header=0)
year = []
bus = []
gdp = []
for index in range(0, len(x_y_data)):
    year.append(x_y_data['Year'][index])
    bus.append(x_y_data["Bus"][index])
    gdp.append(x_y_data["PGDP"][index])

# 进行数据切分,将80用于训练,20用于预测检测
offset = int(len(bus) * 0.8)
X_train = bus[:offset]
X_predict = bus[offset:]
y_train = gdp[:offset]
y_predict = gdp[offset:]
# 一个线性回归模型对象model，此时model内的theta参数并没有值
model = LinearRegression()
X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
X_predict = np.array(X_predict).reshape(-1, 1)
y_predict = np.array(y_predict).reshape(-1, 1)

print(X_train)
print("------")
print(X_predict)
model.fit(X_train, y_train)

# 进行预测
y2 = model.predict(X_predict)
print("预测")
print(y_predict)
print("实际值")
print(y2)


# Plot 为绘图函数，同学们可以利用这个函数建立画布和基本的网格
def Plot():
    plt.figure()
    plt.title('Data')
    plt.xlabel('Diameter(Inches)')
    plt.ylabel('Price(Dollar)')
    plt.axis([np.min(bus), np.max(bus), np.min(gdp), np.max(bus)])
    plt.grid(True)
    return plt


# 绘制网格
plt = Plot()

# 在网格上绘制原始数据散点，图中黑色散点
plt.plot(X_predict, y_predict, 'k.')

# 绘制预测的披萨价格-直径曲线，图中绿色直线
plt.plot(X_predict, y2, 'g-')
plt.show()
