import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sqla
from sklearn.metrics.pairwise import cosine_similarity
import os
import sklearn.tree as tree
import sklearn.model_selection as model_selection
from sklearn import preprocessing
import pydotplus
import seaborn as sns
import sklearn.decomposition as decomposition

os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz 2.44.1/bin'
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("DataMining\\") + len("DataMining\\")]
# 数据来源:阿里天池竞赛平台_天猫真实数据
# 大数据可视化分析作业三:要求分析100000*50的数据
origin_data = pd.read_csv(rootPath + 'Input/mathhomework/car_price/CarPrice_Assignment.csv', header=0)


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
    # 进行数据标准化
    # with_mean 中心化
    # with_std  是否标准化
    # copy 是否复制
    std_origin_data = preprocessing.scale(X=origin_data, with_mean=True, with_std=True, copy=True)
    pd.DataFrame(std_origin_data).to_csv(rootPath + "Output/mathhomework/car_price/std_origin_data.csv")
    cov_data = np.cov(std_origin_data)
    pd.DataFrame(cov_data).to_csv(rootPath + "Output/mathhomework/car_price/cov_data.csv")
    pca = decomposition.PCA(n_components=None, copy=True, whiten=False)
    pca.fit(origin_data)
    # print(pca.explained_variance_ratio_)
    print(pca.get_precision()[0])
    # print(pca.explained_variance_)
    return origin_data


# 一共有21列分别为
# 将数据进行训练和测试的分割
pure_data = ETL(origin_data)
print(pure_data[:2, :])

# criterion：gini,entropy,mse,前者是基尼系数，后者是信息熵
# dt_reg = DecisionTreeRegressor(criterion='entropy',max_depth=24)

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

plt.figure()
plt.xlabel('tree deepth')
plt.ylabel('accuracy rate')
plt.title("deepth-accuracy")
plt.plot(range(len((score * 1000).astype(int))), score, color='#ADFF2F')
plt.xticks(np.arange(1, 25, 1))
plt.show()

# 由图可见当deepth为17的时候,准确率最高
train, test = model_selection.train_test_split(pure_data, test_size=0.3)
dt_reg = tree.DecisionTreeRegressor(criterion='mse', max_depth=17)
dt_reg.fit(train[:, 0:-1], train[:, -1:])

# pip install pydotplus
# pip install graphviz
# 画图方法1-生成dot文件
with open(rootPath + 'Output/mathhomework/car_price/TreeRegressor.dot', 'w') as f:
    dot_data = tree.export_graphviz(dt_reg, out_file=None)
    f.write(dot_data)

    # 画图方法2-生成pdf文件
    dot_data = tree.export_graphviz(dt_reg, out_file=None, feature_names=dt_reg.feature_importances_,
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    ###保存图像到pdf文件
    graph.write_pdf(rootPath + 'Output/mathhomework/car_price/TreeRegressor.pdf')
