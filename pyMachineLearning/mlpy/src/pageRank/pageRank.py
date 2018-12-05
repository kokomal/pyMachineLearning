# coding:utf-8
# Parameter M adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
# sum(i, M_i,j) = 1
# Parameter d damping factor (default value 0.85)
# Parameter eps quadratic error for v (default value 1.0e-8)
# Return v, a vector of ranks such that v_i is the i-th rank from [0, 1]

import numpy as np


# 唯一输入 相邻矩阵
def pagerank(M, eps=1.0e-8, d=0.85):
    N = M.shape[1]
    v = np.random.rand(N, 1)  # N*1
    # print(v)
    norm = np.linalg.norm(v, 1)  # 标准化
    # print(norm)
    v = v / norm
    # print("---NORMALIZED NORM---")
    # print(v)
    last_v = np.ones((N, 1), dtype=np.float32) * 100
    M_hat = (d * M) + (((1 - d) / N) * np.ones((N, N), dtype=np.float32))  # 一次性处理相邻矩阵M即可
    # print(M_hat)
    iter = 1
    while np.linalg.norm(v - last_v, 2) > eps:
        iter += 1
        last_v = v
        v = np.matmul(M_hat, v)
    print("ITER = %d" % iter)
    return v


M = np.array([[0, 0, 0, 0, 1],
              [0.45, 0, 0, 0, 0],
              [0.55, 0, 0, 0, 0],
              [0, 1, 0.5, 0, 0],
              [0, 0, 0.5, 1, 0]])
v = pagerank(M, 0.001, 0.31) # 生物学采用0.31
print(v)
v = pagerank(M, 0.001) # 搜索引擎采用0.85
print(v)
