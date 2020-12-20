import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sqla
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import src.org.wqj.util.mysqlconf as mysqlconf

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("DataMining\\") + len("DataMining\\")]

# 数据来源:阿里天池竞赛平台_天猫真实数据
# 大数据可视化分析作业三:要求分析100000*50的数据
origin_data = pd.read_csv(rootPath + 'Input/mathhomework/car_price/CarPrice_Assignment.csv', header=0)


def ETL(origin_data):
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


# 一共有21列分别为
# 将数据进行训练和测试的分割
pure_data = ETL(origin_data)
print(pure_data[:2, :])
train, test = train_test_split(pure_data, test_size=0.3)
dt_reg = DecisionTreeRegressor(max_depth=24)
dt_reg.fit(train[:, 0:-1], train[:, -1:])
print(dt_reg.score(test[:, 0:-1], test[:, -1:]))
