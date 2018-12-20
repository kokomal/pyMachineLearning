# coding:utf-8
# Parameter M adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
# sum(i, M_i,j) = 1
# Parameter d damping factor (default value 0.85)
# Parameter eps quadratic error for v (default value 1.0e-8)
# Return v, a vector of ranks such that v_i is the i-th rank from [0, 1]

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
    print("M-Hat")
    print(M_hat)
    iter = 1
    while np.linalg.norm(v - last_v, 2) > eps:
        iter += 1
        last_v = v
        v = np.matmul(M_hat, v)
    print("ITER = %d" % iter)
    return v


if __name__ == "__main__":
    # 第一步对M的全0列进行修正，可以是全平摊，也可以是全孤岛
    M = np.array([[0, 0, 0, 0, 1],
                  [0.4277, 1, 0, 0, 0],
                  [0.5723, 0, 0, 0, 0],
                  [0, 0, 0.5, 0, 0],
                  [0, 0, 0.5, 1, 0]])
    v = pagerank(M, 0.001, 0.31)  # 生物学采用0.31
    print(v)
    v = pagerank(M, 0.001)  # 搜索引擎采用0.85,对于孤岛节点，0.85的惩罚系数有点狠了
    print(v)


# 优化版PageRank算法
class PageRank():

    @classmethod
    def fetchFromCSV(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodeset = set() # 节点采用set
        edges = {}
        for line in lines:
            n = line.split(',')
            if not n:
                break
            #nodes[n[0]] = 1
            #nodes[n[2]] = 1
            nodeset.add(n[0])
            nodeset.add(n[2])
            w = 1
            if len(n) == 4:
                w = int(n[3])
            if n[1] == 'in': # caller还是callee
                key = (n[0], n[2])
            else:
                key = (n[2], n[0])
            if key in edges:
                # 这里可以施加数据清洗的操作，例如多次通话的频次、时长的统计学计算等，目前这里仅仅对通话时间进行叠加
                edges[key] += w
            else:
                edges[key] = w
        return cls(nodeset, edges)
        
    def __init__(self, nodeset, edges):
        self.nodeset = nodeset
        self.edges = edges
        self.sorted_node_index_map = {}
        self.sorted_nodes = sorted(self.nodeset)
        i = 0
        for nd in self.sorted_nodes:
            self.sorted_node_index_map[nd] = i
            i += 1
        print(self.sorted_node_index_map)
        # sorted_node_index_map用来给出排序好的{用户:序列}

    def genAdjacentMatrix(self):
        N = len(self.sorted_nodes)
        rawMat = np.eye(N) # 防止全0列
        for edgeTup in self.edges.keys():
            frm = edgeTup[0]
            to = edgeTup[1]
            wt = self.edges[edgeTup]
            frmIdx = self.sorted_node_index_map[frm]
            toIdx = self.sorted_node_index_map[to]
            print('edgeTup', edgeTup)
            print('wt', wt)
            rawMat[toIdx][toIdx] = 0
            rawMat[frmIdx][toIdx] = wt
        print(rawMat) 
        norm = np.linalg.norm(rawMat, axis=0, ord=1)  # 标准化，求得各个列的和
        # print(norm)
        rawMat = rawMat / norm
        return rawMat


    # 简易绘制交互图
    def draw(self, picName):
        G = nx.DiGraph()
        for k in self.edges.keys():
            G.add_edge(k[0], k[1], weight=self.edges[k])
        # nx.draw(G, node_size=300, with_labels=True)
        # nx.draw(G, pos=nx.spring_layout(G), node_color='r', edge_color='g', with_labels=True, font_size=18, node_size=300)
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, node_size=500)
        # 根据权重，实线为权值大的边，虚线为权值小的边
        # edges
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 10]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 10]
        # labels标签定义
        edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        nx.draw_networkx_edges(G, pos, edgelist=elarge,
                            width=4, edge_color='r', arrows=True, arrowsize=6)
        nx.draw_networkx_edges(G, pos, edgelist=esmall,
                            edge_color='b', style='dashed')
        plt.axis('off')
        plt.savefig(picName)
        plt.show()
