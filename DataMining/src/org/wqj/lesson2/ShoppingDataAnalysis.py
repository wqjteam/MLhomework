import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sqla
from sklearn.metrics.pairwise import cosine_similarity
import os

import src.org.wqj.util.mysqlconf as mysqlconf

msqldb = sqla.create_engine('mysql+pymysql://%s:%s@%s/%s?charset=utf8' % (
    mysqlconf.mysql_user, mysqlconf.mysql_password, mysqlconf.mysql_ip, mysqlconf.mysql_database))


def pre_parse_and_load_data():
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find("DataMining\\") + len("DataMining\\")]

    # 数据来源:阿里天池竞赛平台_天猫真实数据
    # 大数据可视化分析作业三:要求分析100000*50的数据
    test_format1 = pd.read_csv(rootPath + 'Input/test_format1.csv', header=0)
    # print(test_format1)
    train_format1 = pd.read_csv(rootPath + 'Input/train_format1.csv', header=0)
    user_info = pd.read_csv(rootPath + 'Input/user_info_format1.csv', header=0)

    # 该数据本是用于机器学习数据,我们将test_format1和train_format1合并
    order = pd.concat([test_format1, train_format1], axis=0)[["merchant_id", "user_id"]]
    print("数据的条数")
    print(order["user_id"].unique().shape[0])  # 有424170条数据
    print("数据的维度")
    print(order["merchant_id"].unique().shape[0])  # 共有1994的维度

    # order_detail = pd.merge(order, user_info, on="user_id", how="left")
    order_detail = pd.merge(order, user_info, left_on=["user_id"], right_on=["user_id"], how="left")
    # 对数据中的空值做清洗
    order_detail.loc[order_detail["age_range"].isna(), ["age_range"]] = 2.0
    order_detail.loc[order_detail["gender"].isna(), ["gender"]] = 1.0
    order_detail[["gender", "age_range"]] = order_detail[["gender", "age_range"]].astype(int)
    print("数据的维度")
    print(order_detail.head(5))

    # 订单详情展示 前条
    #    merchant_id  user_id  age_range  gender
    # 0         4605   163968        0     0
    # 1         1581   360576        2     2
    # 2         1964    98688        6     0
    # 3         3645    98688        6     0
    # 4         3361   295296        2     1
    # 为了使用方便
    # 将数据导入mysql,使用sql做数据分析
    # 在age_range和gender有一部分有空值,直接赋默认值
    order_detail.to_sql("order_detail", msqldb, index=False, if_exists='replace', chunksize=10000)


def anaysis_data():
    # 计算每个用户下单量
    print("用户下单量TOP10")
    purchase_top = pd.read_sql_query("select user_id,purchase_num from ("
                                     "  select user_id,count(merchant_id) as purchase_num from order_detail group by  user_id"
                                     " ) a order by purchase_num desc  limit 10", msqldb)
    print(purchase_top)

    # 找出销售量前10的商品
    print("商品销售量TOP10")
    commodity_top = pd.read_sql_query("select merchant_id,user_num from ("
                                      "  select merchant_id,count(user_id) as user_num from order_detail group by  merchant_id"
                                      " ) a order by user_num desc  limit 10", msqldb)
    print(commodity_top)

    # 各年龄段购买数量
    print("各年龄段购买数量")
    group_age_purchase = pd.read_sql_query(
        "  select age_range,count(1) as purchase_num from order_detail group by  age_range"
        "", msqldb)
    print(group_age_purchase)

    # 展示各性别购买数量
    print("各性别购买数量")
    group_gender_purchase = pd.read_sql_query(
        "  select gender,count(1) as purchase_num from order_detail group by  gender"
        "", msqldb)
    print(group_gender_purchase)

    # 画图
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.xlabel('user_id')
    plt.ylabel('purchase_num')
    plt.title("purchase_top")
    plt.bar(range(len(purchase_top["purchase_num"].values)), purchase_top["purchase_num"].values, color='#ADFF2F',
            tick_label=purchase_top["user_id"].values)
    plt.xticks(rotation=270)
    # 绘制数据标签
    for a, b in zip(range(len(purchase_top["purchase_num"].values)), purchase_top["purchase_num"].values):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    # plt.bar([1,2,3],[10,20,30],fc='r',tick_label=name_list)

    plt.subplot(2, 2, 2)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.xlabel('user_num')
    plt.ylabel('merchant_id')
    plt.title("commodity_top")
    plt.bar(range(len(commodity_top["user_num"].values)), commodity_top["user_num"].values, color='#FFD700',
            tick_label=commodity_top["merchant_id"].values)
    plt.xticks(rotation=270)
    # 绘制数据标签
    # for a, b in zip(range(len(commodity_top["user_num"].values)), commodity_top["user_num"].values):
    #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

    plt.subplot(2, 2, 3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.xlabel('age_range')
    plt.ylabel('purchase_num')
    plt.title("group_age_purchase")
    plt.bar(range(len(group_age_purchase["purchase_num"].values)), group_age_purchase["purchase_num"].values,
            color='#87CEFA',
            tick_label=group_age_purchase["age_range"].values)
    # 绘制数据标签
    for a, b in zip(range(len(group_age_purchase["purchase_num"].values)), group_age_purchase["purchase_num"].values):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

    plt.subplot(2, 2, 4)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.xlabel('gender')
    plt.ylabel('purchase_num')
    plt.title("group_gender_purchase")
    plt.bar(range(len(group_gender_purchase["purchase_num"].values)), group_gender_purchase["purchase_num"].values,
            color='#14c2c3',
            tick_label=group_gender_purchase["gender"].values, width=0.5)
    # 绘制数据标签
    for a, b in zip(range(len(group_gender_purchase["purchase_num"].values)),
                    group_gender_purchase["purchase_num"].values):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

    plt.show()


def recommended_system():
    user_merchant = pd.read_sql_query("  select user_id,merchant_id  from order_detail ", msqldb)
    zero_martix = np.zeros((user_merchant["user_id"].unique().shape[0], user_merchant["merchant_id"].unique().shape[0]))
    # 发现在pandas中聚合数据,会卡死机器,适合放在数据库中跑
    # user_merchants_pandas=user_merchant.groupby('user_id').apply(lambda x: list(x.merchant_id))
    user_merchants = pd.read_sql_query(
        "  select user_id,GROUP_CONCAT(merchant_id) as merchant_ids  from order_detail group by user_id", msqldb)
    merchant_list = user_merchant["merchant_id"].unique().tolist()
    merchant_list.sort(reverse=True)
    user_id_list = user_merchant["user_id"].unique().tolist()
    user_id_list.sort(reverse=True)
    # 组装用户字典
    user_dict = {}
    for i in range(len(user_id_list)):
        user_dict[user_id_list[i]] = i

    # 组装商品字典
    merchant_dict = {}
    for i in range(len(merchant_list)):
        merchant_dict[merchant_list[i]] = i

    # 填写矩阵
    for i in range(user_merchants.shape[0]):
        user_id = user_merchants.iloc[i][0]
        merchant_ids = str(user_merchants.iloc[i][1]).split(",")
        u_id = user_dict[user_id]
        for merchant_id in merchant_ids:
            m_id = merchant_dict[int(merchant_id)]
            zero_martix[u_id][m_id] = 1

    return user_dict, merchant_dict, zero_martix


if __name__ == '__main__':
    # 此次数据分析的目的

    # 1.找出下单量前10的用户
    # 2.找出销售量前10的商品
    # 3.展示各年龄段购买数量
    # 4.展示各性别购买数量
    # 5.基于用户的协同过滤算法,向用户推荐商品

    # 首先预处理数据,将其处理成订单详情数据,共有522341条数据
    # pre_parse_and_load_data()
    # 然后已经有的数据做指标运算并展示
    # anaysis_data()

    # 下面可以做根据用户购买的商品和用户两个基本属性,将数据达成 522341*1994的矩阵
    # 来做基于用户的来做过滤推荐
    user_dict, merchant_dict, zero_martix = recommended_system()

    # 获取到矩阵之后,对用户进行余弦相似度计算即可

    # 比较第一行和第二行的相似度(调用两种方法)
    # 使用np
    score_similar_np = np.dot(zero_martix[1][:], zero_martix[2][:]) / (np.linalg.norm(zero_martix[1][:]) * np.linalg.norm(zero_martix[2][:]))
    # 使用sklearn
    score_similar_sklearn = cosine_similarity(np.mat(zero_martix[1][:]).reshape(1,-1), np.mat(zero_martix[2][:]).reshape(1,-1))
    print(str(score_similar_np)+"||||"+str(score_similar_sklearn))
