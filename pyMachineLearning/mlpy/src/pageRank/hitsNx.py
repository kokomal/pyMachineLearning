# coding:utf-8

import numpy as np

class HITSIterator:
    __doc__ = '''HITS算法,兼顾hub,authorityֵ'''

    def __init__(self, dg):
        self.max_iterations = 100  # 
        self.min_delta = 0.0001  # 
        self.graph = dg

        self.hub = {}
        self.authority = {}
        for node in self.graph.nodes():
            self.hub[node] = 1
            self.authority[node] = 1

    def hits(self):
        if not self.graph:
            return
        flag = False
        for i in range(self.max_iterations):
            change = 0.0  
            norm = 0  #
            tmp = {}
            tmp = self.authority.copy()
            for node in self.graph.nodes():  # 遍历所有节点
                # print(node)
                self.authority[node] = 0  # 每一个节点的authority先设置为0
                # for incident_page in self.graph.incidents(node):  # 遍历此节点的所有入度，然后authority再更新为入度的hub之和
                for incident_page in self.graph.in_edges(node):  # 遍历此节点的所有入度，然后authority再更新为入度的hub之和
                    # print(incident_page[0])
                    self.authority[node] += self.hub[incident_page[0]]
                norm += pow(self.authority[node], 2)  # norm均一化为√auth1^2+auth2^2+...
            norm = np.sqrt(norm)
            for node in self.graph.nodes():  # 再度遍历所有节点
                self.authority[node] /= norm  # authority进行均一化
                change += abs(tmp[node] - self.authority[node])
            norm = 0
            tmp = self.hub.copy()
            for node in self.graph.nodes():  # 第三次遍历所有节点
                self.hub[node] = 0  # 每一个节点的hub先设置为0
                # for neighbor_page in self.graph.neighbors(node):  # 遍历此节点的所有出度邻居
                for neighbor_page in self.graph.out_edges(node):  # 遍历此节点的所有出度邻居
                    self.hub[node] += self.authority[neighbor_page[1]]  # hub为邻居的authority之和
                norm += pow(self.hub[node], 2)  # norm均一化为√hub1^2+hub2^2+...
            norm = np.sqrt(norm)
            for node in self.graph.nodes():
                self.hub[node] /= norm
                change += abs(tmp[node] - self.hub[node])
            print("This is NO.%s iteration" % (i + 1))
            print("authority", self.authority)
            print("hub", self.hub)

            if change < self.min_delta:
                flag = True
                break
        if flag:
            print("finished in %s iterations!" % (i + 1))
        else:
            print("finished out of 100 iterations!")

        print("The best authority page: ", max(self.authority.items(), key=lambda x: x[1]))
        print("The best hub page: ", max(self.hub.items(), key=lambda x: x[1]))


if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt
    dg = nx.DiGraph()
    # dg = digraph()

    # dg.add_nodes(["A", "B", "C", "D", "E"])

    dg.add_nodes_from(["A", "B", "C", "D", "E"])

    dg.add_edge("A", "C")
    dg.add_edge("A", "D")
    dg.add_edge("B", "D")
    dg.add_edge("C", "E")
    dg.add_edge("D", "E")
    dg.add_edge("B", "E")
    dg.add_edge("E", "A")

    nx.draw(dg, node_size=300, with_labels=True)
    plt.savefig("hits.png")
    plt.show()
    hits = HITSIterator(dg)
    hits.hits()
