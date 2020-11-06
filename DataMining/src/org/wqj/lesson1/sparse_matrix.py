import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt




# 生成随机稀疏矩阵
num_col = 20
num_row = 10
num_ele = 40
a = [np.random.randint(0,num_row) for _ in range(num_ele)]
b = [np.random.randint(0,num_col) for _ in range(num_ele-num_col)] + [i for i in range(num_col)]  # 保证每一列都有值，不会出现全零列
c = [np.random.rand() for _ in range(num_ele)]
rows, cols, v = np.array(a), np.array(b), np.array(c)

sparseX = ss.coo_matrix((v,(rows,cols)))
X = sparseX.todense()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(X, vmin=0, vmax=1, cmap='magma')
ax.set_xticks([])
ax.set_yticks([])
plt.show()