import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sqla
from sklearn.metrics.pairwise import cosine_similarity
import os

from sklearn.model_selection import train_test_split

import src.org.wqj.util.mysqlconf as mysqlconf

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("DataMining\\") + len("DataMining\\")]

# 数据来源:阿里天池竞赛平台_天猫真实数据
# 大数据可视化分析作业三:要求分析100000*50的数据
origin_data = pd.read_csv(rootPath + 'Input/mathhomework/car_price/CarPrice_Assignment.csv', header=0)
print(origin_data[:10][:])
origin_data.as_matrix()
#一共有21列分别为
#将数据进行训练和测试的分割

train,test=train_test_split(origin_data,test_size=0.3)