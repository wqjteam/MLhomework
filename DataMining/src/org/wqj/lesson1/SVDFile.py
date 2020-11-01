# coding=utf-8
import numpy as np
import os
import scipy.sparse as ss
import xlwt
from numpy import linalg as la
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("DataMining\\") + len("DataMining\\")]
dataPath = rootPath + "Input/"
col = [0] * 500
row = [0] * 500
data = [0] * 500
for i in range(500):
    row[i] = np.random.randint(0, 50)
    col[i] = np.random.randint(0, 200)
    data[i] = np.random.randint(1,11)
# print(max(col))
spr_A = ss.coo_matrix((data, (row, col)), shape=(100, 100)).reshape(50,200).toarray()  # 构造50*200的矩阵
# 利用excel保存
f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
for c in range(200):
    for r in range(50):
        sheet1.write(r, c, int(spr_A[r, c]))  # 注意要使用int(spr_A[r,c])，直接使用spr_A[r,c]会报错
f.save(dataPath+'text.xls')

print(spr_A.shape)
# 调用基本库SVD变换：
u, s, v = la.svd(spr_A, full_matrices=0, compute_uv=1)
# 返回出原始矩阵
print("U：")
print(u, u.shape)
print("S：")
print(s, s.shape)
print("V：")
print(v, v.shape)

