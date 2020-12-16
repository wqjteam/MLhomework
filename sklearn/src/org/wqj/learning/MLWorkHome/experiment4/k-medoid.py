from numpy import *
import numpy as np
import time
import matplotlib.pyplot as plt
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("sklearn\\") + len("sklearn\\")]
dataPath = rootPath + "Input/MLWorkHome/experiment4/Data.txt"


# euclDistance函数计算两个向量之间的欧氏距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# initCentroids选取任意数据集中任意样本点作为初始均值点
# dataSet: 数据集， k: 人为设定的聚类簇数目
# centroids： 随机选取的初始均值点
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# kmeans: k-means聚类功能主函数
# 输入：dataSet-数据集，k-人为设定的聚类簇数目
# 输出：centroids-k个聚类簇的均值点，clusterAssment－聚类簇中的数据点
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]

    clusterAssment = mat(zeros((numSamples, 2)))
    # clusterAssment第一列存储当前点所在的簇
    # clusterAssment第二列存储点与质心点的距离
    clusterChanged = True

    ## 步骤一: 初始化均值点
    centroids = initCentroids(dataSet, k)
    runcount = 0
    while clusterChanged:
        runcount += 1
        clusterChanged = False
        ## 遍历每一个样本点
        for i in range(numSamples):
            # minDist：最近距离
            # minIndex：最近的均值点编号
            minDist = 100000.0
            minIndex = 0
            ## 步骤二: 寻找最近的均值点
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            ## 步骤三: 更新所属簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## 步骤四: 更新簇的均值点
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            # nonzero(clusterAssment[:, 0].A == j) 的到数据中不等于j的索引[i][j]
            # clusterAssment和dataSet 是一一对应的 (数据条目相同)
            # centroids[j, :] = mean(pointsInCluster, axis=0)

            centroids[j, :] =k_medoid(pointsInCluster)
            #centroids[j, :] = np.median(pointsInCluster, axis=0)
        if (runcount % 10 == 0):
            print('运行次数:%d' % runcount)
    print('Congratulations, cluster complete!')
    return centroids, clusterAssment


def k_medoid(pointsInCluster):
    min_dist_sum = 1000000
    min_dist_index = 0
    # 获取到聚簇中距离所有最近的点，即可
    for pointsInCluster_index in range(pointsInCluster.shape[0]):
        dist_sum = 0
        for pointsInCluster_index_2 in range(pointsInCluster.shape[0]):
            dist_sum += euclDistance(pointsInCluster[pointsInCluster_index, :]
                                     , pointsInCluster[pointsInCluster_index_2, :])
        if dist_sum < min_dist_sum:
            min_dist_index = pointsInCluster_index
            min_dist_sum = dist_sum
    #dataSet[pointsInCluster]写错了
    return pointsInCluster[min_dist_index, :]


# showCluster利用pyplot绘图显示聚类结果（二维平面）
# 输入:dataSet-数据集，k-聚类簇数目，centroids-聚类簇的均值点，clusterAssment－聚类簇中数据点
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry, the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        return 1

    # 画出所有的样本点
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 标记簇的质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


## step 1: 载入数据
dataSet = []
fileIn = open(dataPath)
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

## step 2: 开始聚类...
dataSet = mat(dataSet)
k = 4
centroids, clusterAssment = kmeans(dataSet, k)

## step 3: 显示聚类结果
showCluster(dataSet, k, centroids, clusterAssment)
