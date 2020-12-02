import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sqla
import os

ip="115.159.151.166"
database="study"
user="root"
password="buaaai123456"



curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("DataMining\\") + len("DataMining\\")]

# 数据来源:阿里天池天猫真实数据
# 大数据可视化分析作业三:要求分析100000*50的数据
test_format1 = pd.read_csv(rootPath + 'Input/test_format1.csv', header=0)
# print(test_format1)
train_format1 = pd.read_csv(rootPath + 'Input/train_format1.csv', header=0)
user_info = pd.read_csv(rootPath + 'Input/user_info_format1.csv', header=0)

# 该数据本是用于机器学习数据,我们将test_format1和train_format1合并
order = pd.concat([test_format1, train_format1], axis=0)[["merchant_id", "user_id"]]
print("数据的条数")
print(order["user_id"].count())  # 有522341条数据
print("数据的维度")
print(order["merchant_id"].unique().shape[0])  # 共有1994的维度

# order_detail = pd.merge(order, user_info, on="user_id", how="left")
order_detail = pd.merge(order, user_info, left_on=["user_id"], right_on=["user_id"], how="left")

print("数据的维度")
print(order_detail.head(5))

msqldb = sqla.create_engine('mysql+pymysql://%s:%s@%s/%s?charset=utf8'%(user,password,ip,database))
# 订单详情展示 前条
#    merchant_id  user_id  age_range  gender
# 0         4605   163968        0.0     0.0
# 1         1581   360576        2.0     2.0
# 2         1964    98688        6.0     0.0
# 3         3645    98688        6.0     0.0
# 4         3361   295296        2.0     1.0

# 此次数据分析的目的

# 1.找出下单量前10的用户
# 2.找出销售量前10的商品
# 3.展示各年龄段购买数量
# 4.展示各性别购买数量
# 5.基于用户的协同过滤算法,向用户推荐商品


# 为了使用方便
# 将数据导入mysql,使用sql做数据分析
# 在age_range和gender有一部分有空值,直接赋默认值
# order_detail.to_sql("order_detail",msqldb, index=False, if_exists='replace',chunksize=10000)


#计算每个用户下单量
print("用户下单量TOP10")
purchase_top = pd.read_sql_query("select user_id,purchase_num from ("
                                 "  select user_id,count(merchant_id) as purchase_num from order_detail group by  user_id"
                                 " ) a order by purchase_num desc  limit 10", msqldb)
print(purchase_top)

#找出销售量前10的商品
print("商品销售量TOP10")
commodity_top = pd.read_sql_query("select merchant_id,user_num from ("
                                  "  select merchant_id,count(user_id) as user_num from order_detail group by  merchant_id"
                                  " ) a order by user_num desc  limit 10", msqldb)
print(commodity_top)


#各年龄段购买数量
print("各年龄段购买数量")
group_age_purchase = pd.read_sql_query("  select age_range,count(1) as purchase_num from order_detail group by  age_range"
                                       "", msqldb)
print(group_age_purchase)

#展示各性别购买数量
print("各性别购买数量")
group_gender_purchase = pd.read_sql_query("  select gender,count(1) as purchase_num from order_detail group by  gender"
                                       "", msqldb)
print(group_gender_purchase)


#画图
plt.figure()
plt.subplot(2, 2, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.xlabel('Height(m)')
plt.ylabel('Weight(kg)')
plt.title("purchase_top")
name_list = ['Monday','Tuesday','Friday']
plt.bar([1,2,3],[10,20,30],fc='r')
# plt.bar([1,2,3],[10,20,30],fc='r',tick_label=name_list)
plt.show()
#下面可以做根据用户购买的商品和用户两个基本属性
# 来做基于用户的来做过滤推荐
