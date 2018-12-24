# coding:utf-8
"""
@summary: LeaderRank
@attention: SpecialThanksTo 吕琳媛给出的比PageRank更加高效稳定的算法
"""
import numpy as np
eps = 1e-3


# LeaderRank算法
class LeaderRank():

    @classmethod
    def fetchFromCSV(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodeset = set()  # 节点采用set
        edges = {}
        for line in lines:
            n = line.split(',')
            if not n:
                break
            nodeset.add(n[0])
            nodeset.add(n[2])
            w = 1
            if len(n) == 4:
                w = int(n[3])
            if n[1] == 'in':  # caller还是callee
                key = (n[0], n[2])
            else:
                key = (n[2], n[0])
            if key in edges:
                # 这里可以施加数据清洗的操作，例如多次通话的频次、时长的统计学计算等，目前这里仅仅对通话时间进行叠加
                edges[key] += w
                # edges[key] = 1
            else:
                edges[key] = w
                # edges[key] = 1
        return cls(nodeset, edges)
    
    def __init__(self, nodeset, edges):
        self.N = len(nodeset)
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
        N = self.N
        rawMat = np.zeros((N + 1, N + 1))  # N+1的维度了，考虑了Gnode
        for edgeTup in self.edges.keys():
            frm = edgeTup[0]
            to = edgeTup[1]
            wt = self.edges[edgeTup]
            frmIdx = self.sorted_node_index_map[frm]
            toIdx = self.sorted_node_index_map[to]
            rawMat[toIdx][toIdx] = 0
            rawMat[frmIdx][toIdx] = wt
        rawMatT = rawMat.T.copy()
        # 这里是LeaderRank额外的一步，即添加Gnode
        for i in range(N):
            rawMat[i][N] = 1  # if sm == 0 else sm # TODO 是否防止全0？
            sm = sum(rawMat[:, i])
            rawMat[N][i] = sm
            
            rawMatT[i][N] = 1
            sm = sum(rawMatT[:, i])
            rawMatT[N][i] = sm

        return rawMat, rawMatT

    def prep(self, rawMat):
        self.sw_j = {}  # 出度权重Ewjl,j = 1 to N+1
        for j in range(self.N + 1):
            self.sw_j[j] = sum(rawMat[j, :])
        print('出度', self.sw_j)

    def leaderRank(self, callerOrCallee='caller',):
        rawMat, rawMatT = self.genAdjacentMatrix()
        if callerOrCallee == 'caller':
            M = rawMat
        else:
            M = rawMatT
        self.prep(M)
        print(M)
        LR = dict.fromkeys(range(self.N + 1), 1.0)
        LR[self.N] = 0.0
        # print('LR', LR)
        iter = 0
        while True:
            iter += 1
            tempLR = {}
            for i in range(self.N + 1):
                s = 0.0
                for j in range(self.N + 1):
                    if M[j][i]:
                        s += LR[j] * M[j][i] / self.sw_j[j]
                # print('s', s, 'I=', i)
                tempLR[i] = s
            # 终止条件:LR值不在变化
            # print('tmp', tempLR)
            error = 0
            for n in tempLR.keys():
                error += abs(tempLR[n] - LR[n])
            if error <= eps:
                break
            LR = tempLR
        print("ITER=" , iter)
        avg = LR[self.N] / self.N
        print(avg)
        LR.pop(self.N)
        for k in LR.keys():
            LR[k] += avg
        print(LR)
        for nd in self.sorted_node_index_map.keys():
            print("USER-%s Rank is %f" % (nd, LR[self.sorted_node_index_map[nd]]))
