import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sqla
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("DataMining\\") + len("DataMining\\")]

# 数据来源:阿里天池天猫真实数据
# 大数据可视化分析作业三:要求分析100000*50的数据
test_format1 = pd.read_csv(rootPath + 'Input/test_format1.csv', header=0)
# print(test_format1)
train_format1 = pd.read_csv(rootPath + 'Input/train_format1.csv', header=0)
user_info = pd.read_csv(rootPath + 'Input/user_info_format1.csv', header=0)

# 该数据本是用于机器学习数据,我们将test_format1和train_format1合并
order = pd.concat([test_format1, train_format1],axis=0)[["merchant_id", "user_id"]]
print("数据的条数")
print(order["user_id"].count())  # 有522341条数据
print("数据的维度")
print(order["merchant_id"].unique().shape[0])  # 共有1994的维度

# order_detail = pd.merge(order, user_info, on="user_id", how="left")
order_detail = pd.merge(order, user_info, left_on=["user_id"], right_on=["user_id"], how="left")

print("数据的维度")
print(order_detail.head(5))

msqldb=sqla.create_engine('mysql+pymysql://root:buaaai123456@115.159.151.166/study?charset=utf8')
#订单详情展示 前条
#    merchant_id  user_id  age_range  gender
# 0         4605   163968        0.0     0.0
# 1         1581   360576        2.0     2.0
# 2         1964    98688        6.0     0.0
# 3         3645    98688        6.0     0.0
# 4         3361   295296        2.0     1.0

#此次数据分析的目的

#1.找出下单量前10的用户
#2.找出被购买前10的商品
#3.展示各年龄段购买数量
#4.展示各性别购买数量
#5.基于用户的协同过滤算法,向用户推荐商品


#为了使用方便
#将数据导入mysql,使用sql做数据分析
order_detail.head(10).to_sql("order_detail",msqldb, index=False, if_exists='replace',chunksize=10000)
print("计算每个用户下单量")
# a,b=order_detail.groupby('user_id').size()
# new=pd.DataFrame(order_detail.groupby('user_id').size())
# print(new.head(10))

pd.read_sql_query("select * from order_detail limit 10",msqldb)