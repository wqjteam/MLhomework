from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.tree as tree
import sklearn.model_selection as model_selection
from sklearn import preprocessing, metrics
import pydotplus
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import numpy.linalg as nlg
from sklearn.externals import joblib

os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz 2.44.1/bin'
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("DataMining\\") + len("DataMining\\")]
# 数据来源:阿里天池竞赛平台_天猫真实数据
# 大数据可视化分析作业三:要求分析100000*50的数据
origin_data = pd.read_csv(rootPath + 'Input/mathhomework/car_price/CarPrice_Assignment.csv', header=0)
origin_data_columns = origin_data.columns


def ETL(origin_data):
    # pd.plotting.scatter_matrix(origin_data, alpha=0.7, figsize=(15, 15), diagonal='kde')
    # plt.show()
    # print(origin_data.corr())
    # annot: 默认为False，为True的话，会在格子上显示数字
    # sns.heatmap(origin_data.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=False)
    sns.heatmap((origin_data.drop('car_ID', axis=1)).corr(), vmin=-1, vmax=1, square=True, annot=False)
    origin_data = np.mat(origin_data)
    for index in range(origin_data.shape[0]):
        origin_data[index, 2] = str(origin_data[index, 2]).split(" ")[0].lower()
    keys = np.unique(origin_data[:, 2].A)
    # print(keys)
    values = np.arange(1, len(keys) + 1, 1)
    car_brand_dict = dict(zip(keys, values))
    car_fueltype_dict = {"gas": 1, "diesel": 2}
    car_aspiration_dict = {"std": 1, "turbo": 2}
    car_doornumber_dict = {"two": 2, "four": 4}
    car_carbody_dict = {"sedan": 1, "hatchback": 2,
                        "wagon": 3, "hardtop": 4, "convertible": 5}
    car_drivewheel_dict = {"fwd": 1, "rwd": 2, "4wd": 3}
    car_enginelocation_dict = {"front": 1, "rear": 2}
    car_enginetype_dict = {"ohc": 1, "ohcf": 2, "ohcv": 3, "dohc": 4, "l": 5, "rotor": 6, "dohcv": 7}
    car_cylindernumber_dict = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}
    car_fuelsystem_dict = {"mpfi": 1, "2bbl": 2, "idi": 3, "1bbl": 4, "spdi": 5, "4bbl": 6, "spfi": 7, "mfi": 8}
    # 对品牌进行处理
    for key, val in zip(keys, values):
        origin_data[np.nonzero(origin_data[:, 2].A == key)[0], 2] = val
    # 对燃油类型处理
    for k in car_fueltype_dict:
        origin_data[np.nonzero(origin_data[:, 3].A == k)[0], 3] = car_fueltype_dict[k]
    for k in car_aspiration_dict:
        origin_data[np.nonzero(origin_data[:, 4].A == k)[0], 4] = car_aspiration_dict[k]
    for k in car_doornumber_dict:
        origin_data[np.nonzero(origin_data[:, 5].A == k)[0], 5] = car_doornumber_dict[k]
    for k in car_carbody_dict:
        origin_data[np.nonzero(origin_data[:, 6].A == k)[0], 6] = car_carbody_dict[k]
    for k in car_drivewheel_dict:
        origin_data[np.nonzero(origin_data[:, 7].A == k)[0], 7] = car_drivewheel_dict[k]
    for k in car_enginelocation_dict:
        origin_data[np.nonzero(origin_data[:, 8].A == k)[0], 8] = car_enginelocation_dict[k]
    for k in car_enginetype_dict:
        origin_data[np.nonzero(origin_data[:, 14].A == k)[0], 14] = car_enginetype_dict[k]
    for k in car_cylindernumber_dict:
        origin_data[np.nonzero(origin_data[:, 15].A == k)[0], 15] = car_cylindernumber_dict[k]
    for k in car_fuelsystem_dict:
        origin_data[np.nonzero(origin_data[:, 17].A == k)[0], 17] = car_fuelsystem_dict[k]
    # 移除自增长id
    origin_data = np.delete(origin_data, 0, axis=1)

    return origin_data


def PCA(origin_data):
    # 进行数据标准化
    # with_mean 中心化
    # with_std  是否标准化
    # copy 是否复制
    origin_data = pd.DataFrame(origin_data)
    columns_list = list(origin_data_columns.values)
    columns_list.pop(0)
    std_origin_data = preprocessing.scale(X=origin_data, with_mean=True, with_std=True, copy=True)
    std_df_containt_price = pd.DataFrame(std_origin_data)
    std_df_containt_price.columns = columns_list
    std_df = std_df_containt_price.drop(['price'], axis=1)
    std_price = std_df_containt_price['price']
    std_df.to_csv(rootPath + "Output/mathhomework/car_price/std_origin_data.csv")
    cov_data = std_df.corr()

    eig_value, eig_vector = nlg.eig(cov_data)
    eig = pd.DataFrame()  # 利用变量名和特征值建立一个数据框
    eig['names'] = cov_data.columns  # 列名
    eig['eig_value'] = eig_value  # 特征值
    # pd.DataFrame(cov_data).to_csv(rootPath + "Output/mathhomework/car_price/cov_data.csv")
    for k in range(1, 24):  # 确定公共因子个数
        if eig['eig_value'][:k].sum() / eig['eig_value'].sum() >= 0.85:  # 如果解释度达到00%, 结束循环
            print(k)
            break
    # k为9，前9个特征值提供了80的贡献率
    A = pd.DataFrame([sqrt(eig_value[i]) * eig_vector[:, i] for i in range(9)]).T  # 构造因子载荷矩阵A
    A.columns = ['factor%d' % (i + 1) for i in range(9)]  # 因子载荷矩阵A的公共因子
    h = np.zeros(21)  # 变量共同度，反映变量对共同因子的依赖程度，越接近1，说明公共因子解释程度越高，因子分析效果越好
    D = np.mat(np.eye(21))  # 特殊因子方差，因子的方差贡献度 ，反映公共因子对变量的贡献，衡量公共因子的相对重要性
    A = np.mat(A)  # 将因子载荷阵A矩阵化
    for i in range(21):
        a = A[i, :] * A[i, :].T  # A的元的行平方和
        h[i] = a[0, 0]  # 计算变量X共同度,描述全部公共因子F对变量X_i的总方差所做的贡献，及变量X_i方差中能够被全体因子解释的部分
        D[i, i] = 1 - a[0, 0]  # 因为自变量矩阵已经标准化后的方差为1，即Var(X_i)=第i个共同度h_i + 第i个特殊因子方差
    rotation_mat = varimax(A)  # 调用方差最大旋转函数
    rotation_mat = pd.DataFrame(rotation_mat)  # 数据框化
    rotation_mat.columns = ['factor%d' % (i + 1) for i in range(9)]
    # 已旋转
    columns_list.pop(-1)
    rotation_mat.index = columns_list
    rotation_mat.to_csv(rootPath + "Output/mathhomework/car_price/loadfactormatrix .csv")


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):  # 定义方差最大旋转函数
    p, k = Phi.shape  # 给出矩阵Phi的总行数，总列数
    R = np.eye(k)  # 给定一个k*k的单位矩阵
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)  # 矩阵乘法
        u, s, vh = nlg.svd(np.dot(Phi.T, np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(
            np.diag(np.dot(Lambda.T, Lambda))))))  # 奇异值分解svd
        R = np.dot(u, vh)  # 构造正交矩阵R
        d = sum(s)  # 奇异值求和
        if d_old != 0 and d / d_old < 1 + tol: break
    return np.dot(Phi, R)  # 返回旋转矩阵Phi*R


# 一共有25列分别为
# 将数据进行训练和测试的分割
pure_data = ETL(origin_data)
PCA(pure_data)
print(pure_data[:2, :])


# 对决策树深度进行判断
score = np.empty((1, 24), dtype=float)

score = score.flatten()
for i in range(24):
    for j in range(10):
        train, test = model_selection.train_test_split(pure_data, test_size=0.3)
        dt_reg = tree.DecisionTreeRegressor(criterion='mse', max_depth=i + 1)
        dt_reg.fit(train[:, 0:-1], train[:, -1:])
        score[i] += dt_reg.score(test[:, 0:-1], test[:, -1:])
    score[i] = float('%.3f' % (score[i] / 10))


# 由图可见当deepth为17的时候,准确率最高
train, test = model_selection.train_test_split(pure_data, test_size=0.3)
dt_reg = tree.DecisionTreeRegressor(criterion='mse', max_depth=17)
dt_reg.fit(train[:, 0:-1], train[:, -1:])

model = RandomForestRegressor(n_estimators=100,max_depth=24)
model.fit(train[:, 0:-1], train[:, -1:])
predicted = model.predict(test[:, 0:-1])
# 生成pkl文件，供后期调用
# joblib.dump(model, 'RF.pkl')


plt.figure()
# plt.xlabel('tree deepth')
# plt.ylabel('accuracy rate')
plt.title("deepth-accuracy")
plt.plot(range(len(predicted)), predicted, color='#ADFF2F')
plt.scatter(range(len(test[:, -1:])), np.array(test[:, -1:].flatten()))
# plt.xticks(np.arange(1, 25, 1))
plt.show()

# pip install pydotplus
# pip install graphviz
# 画图方法1-生成dot文件
# with open(rootPath + 'Output/mathhomework/car_price/TreeRegressor.dot', 'w') as f:
#     dot_data = tree.export_graphviz(dt_reg, out_file=None)
#     f.write(dot_data)
#
#     # 画图方法2-生成pdf文件
#     dot_data = tree.export_graphviz(dt_reg, out_file=None, feature_names=dt_reg.feature_importances_,
#                                     filled=True, rounded=True, special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     ###保存图像到pdf文件
#     graph.write_pdf(rootPath + 'Output/mathhomework/car_price/TreeRegressor.pdf')
