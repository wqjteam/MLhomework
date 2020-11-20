# coding=utf-8
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# 生成的随机想按照seed(0)的下标一样,相同
np.random.seed(0)
# 生成随机的两类数据点，每类中各20个样本点
# x为样本点的横坐标，y为样本点的纵坐标
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# 此为颜色,进行分类,前20个后20个
y = [0] * 20 + [1] * 20

# 利用plt展示随机生成的两类数据点
plt.figure(figsize=(8, 4))
# [:,0]取x,[:,1]取y
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
plt.axis('tight')
plt.show()

# 训练模型，核函数为线性函数，其余参数默认
clf = svm.SVC(kernel='linear')
# 利用训练的模型拟合数据样本点
clf.fit(x, y)

# 获取参数w列表
w = clf.coef_[0]
# a为最优分类线的斜率
a = -w[0] / w[1]

# 绘制SVM分类后的效果
# xx为待绘制的分类线区间，(-5,5)之间x的值
xx = np.linspace(-5, 5)
# yy为分类线区间上对应x的y值
# intercept_[0]/w[1]为截距
yy = a * xx + -(clf.intercept_[0]) / w[1]  ##yy分类线

# 画出与支持向量点相切的线
# clf.support_vectors_为支持向量点的坐标
# b首先为分类线下方的一个支持向量点(support_vectors切点)
b = clf.support_vectors_[0]
# yy_down为分类线下方的支持向量点的纵坐标
yy_down = a * xx + (b[1] - a * b[0])
# b此时为分类线上方的一个支持向量点(-1的下标为最大的坐标)
b2 = clf.support_vectors_[-1]
# yy_up为分类线上方的支持向量点的纵坐标
yy_up = a * xx + (b2[1] - a * b2[0])

# 最优分类线的w参数
print("W:", w)
# 最优分类线的斜率
print("a:", a)

# 基于svm拟合的模型中，支持向量点的坐标
print("support_vectors_:", clf.support_vectors_)
# 基于svm拟合的模型中，x和y的系数
print("clf.coef_:", clf.coef_)

plt.figure(figsize=(8, 4))
plt.plot(xx, yy)
plt.plot(xx, yy_down)
plt.plot(xx, yy_up)
#s代表元素的大小
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)

plt.axis('tight')

plt.show()
