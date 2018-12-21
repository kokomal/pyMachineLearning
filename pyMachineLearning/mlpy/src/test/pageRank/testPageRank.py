# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from knn import knn
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx  # 导入networkx包
from pageRank.pageRank import PageRank


# 测试逻辑回归
class TestPageRank(unittest.TestCase):  # 继承unittest.TestCase

    def tearDown(self):
        # 每个测试用例执行之后做操作
        print('***AFTER ONE TEST***')

    def setUp(self):
        # 每个测试用例执行之前做操作
        print('***BEFORE ONE TEST***')

    @classmethod
    def tearDownClass(self):
    # 必须使用 @ classmethod装饰器, 所有test运行完后运行一次
         print('FINISHED...')

    @classmethod
    def setUpClass(self):
    # 必须使用@classmethod 装饰器,所有test运行前运行一次
        print('STARTING...')
    
    def test_npArray(self):
        import numpy as np
        x = np.array([[0, 3, 4],
                      [1, 6, 4]])
        # 默认参数ord=None，axis=None，keepdims=False
        print ("默认参数(矩阵2范数，不保留矩阵二维特性)：", np.linalg.norm(x))
        print ("矩阵2范数，保留矩阵二维特性：", np.linalg.norm(x, keepdims=True))
         
        print ("矩阵每个行向量求向量的2范数：", np.linalg.norm(x, axis=1, keepdims=True))
        print ("矩阵每个列向量求向量的2范数：", np.linalg.norm(x, axis=0, keepdims=True))
         
        print ("矩阵1范数：", np.linalg.norm(x, ord=1, keepdims=True))
        print ("矩阵2范数：", np.linalg.norm(x, ord=2, keepdims=True))
        print ("矩阵∞范数：", np.linalg.norm(x, ord=np.inf, keepdims=True))
         
        print ("矩阵每个行向量求向量的1范数：", np.linalg.norm(x, ord=1, axis=1, keepdims=True))
        print ("矩阵每个行向量求向量的1范数：", np.linalg.norm(x, ord=1, axis=0, keepdims=True))
     
    def test_npNorm(self):
        import numpy as np
        v = np.array([[0, 1, 2],
                      [3, 4, 5]])
        last_v = np.array([[0, 1, 2],
                           [6, 8, 5]])
        eps = np.linalg.norm(v - last_v, 2)  # √x1^2+x2^2+...
        print(eps)
    
    def test_dict(self):
        dic = {'a':31, 'bc':5, 'c':3, 'asd':4, 'aa':74, 'd':0}
        sorted_dict = sorted(dic.items(), key=lambda d:d[0]) 
        print(dic)
        print(sorted_dict)
        print(type(dic))
        print(type(sorted_dict))

    def test_draw(self):
        # !-*- coding:utf8-*-
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        G.add_node(1)
        G.add_node(2)
        G.add_nodes_from([3, 4, 5, 6])
        G.add_cycle([1, 2, 3, 4])
        G.add_edge(1, 3, weight=10)
        G.add_edges_from([(3, 5), (3, 6), (6, 7)])
        nx.draw(G, node_size=300, with_labels=True)
        plt.savefig("youxiangtu.png")
        plt.show()
        
    def test_draw2(self):
        G = nx.random_graphs.barabasi_albert_graph(100, 1)  # 生成一个BA无标度网络G
        nx.draw(G, node_size=300, with_labels=True)
        plt.savefig("ba.png")  # 输出方式1: 将图像存为一个png格式的图片文件
        plt.show()  # 输出方式2: 在窗口中显示这幅图像

    def test_draw3(self):
        __author__ = """Aric Hagberg (hagberg@lanl.gov)"""
        try:
            import matplotlib.pyplot as plt
        except:
            raise
        import networkx as nx
         
        G = nx.Graph()
        # 添加带权边
        G.add_edge('a', 'b', weight=0.6)
        G.add_edge('a', 'c', weight=0.2)
        G.add_edge('c', 'd', weight=0.1)
        G.add_edge('c', 'e', weight=0.7)
        G.add_edge('c', 'f', weight=0.9)
        G.add_edge('a', 'd', weight=0.3)
        # 按权重划分为重权值得边和轻权值的边
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
        # 节点位置
        pos = nx.shell_layout(G)  # positions for all nodes
        # 首先画出节点位置
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)
        # 根据权重，实线为权值大的边，虚线为权值小的边
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge,
                            width=10)
        nx.draw_networkx_edges(G, pos, edgelist=esmall,
                            width=2, alpha=0.5, edge_color='b', style='dashed')
         
        # labels标签定义
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
         
        plt.axis('off')
        plt.savefig("weighted_graph.png")  # save as png
        plt.show()  # display

    def test_draw4(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        G.add_edges_from(
            [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
             ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])
        
        val_map = {'A': 1.0,
                   'D': 0.5714285714285714,
                   'H': 0.0}
        
        values = [val_map.get(node, 0.25) for node in G.nodes()]
        
        # Specify the edges you want here
        red_edges = [('A', 'C'), ('E', 'C')]
        edge_colours = ['black' if not edge in red_edges else 'red'
                        for edge in G.edges()]
        black_edges = [edge for edge in G.edges() if edge not in red_edges]
        
        # Need to create a layout when doing
        # separate calls to draw nodes and edges
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                               node_color=values, node_size=500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
        plt.show()

    def test_draw5(self):
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt
        import pylab
        
        G = nx.DiGraph()
        
        G.add_edges_from([('A', 'B'), ('C', 'D'), ('G', 'D')], weight=1)
        G.add_edges_from([('D', 'A'), ('D', 'E'), ('B', 'D'), ('D', 'E')], weight=2)
        G.add_edges_from([('B', 'C'), ('E', 'F')], weight=3)
        G.add_edges_from([('C', 'F')], weight=4)
        
        val_map = {'A': 1.0,
                           'D': 0.5714285714285714,
                                      'H': 0.0}
        
        values = [val_map.get(node, 0.45) for node in G.nodes()]
        edge_labels = dict([((u, v,), d['weight'])
                         for u, v, d in G.edges(data=True)])
        red_edges = [('C', 'D'), ('D', 'A')]
        edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
        
        pos = nx.spring_layout(G)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw(G, pos, node_color=values, node_size=1500, edge_color=edge_colors, edge_cmap=plt.cm.Reds)
        pylab.show()
    
    def test_draw6(self):
        # coding=utf-8
        import numpy as np
        import matplotlib.pyplot as plt

        def height(x, y):
            return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

        x = np.linspace(-3, 3, 300)
        y = np.linspace(-3, 3, 300)
        X, Y = np.meshgrid(x, y)
        plt.contourf(X, Y, height(X, Y), 10, alpha=0.75, cmap=plt.cm.hot)
        # 为等高线填充颜色 10表示按照高度分成10层 alpha表示透明度 cmap表示渐变标准
        C = plt.contour(X, Y, height(X, Y), 10, colors='black')
        # 使用contour绘制等高线
        plt.clabel(C, inline=True, fontsize=10)
        # 在等高线处添加数字
        plt.xticks(())
        plt.yticks(())
        # 去掉坐标轴刻度
        plt.show()
        # 显示图片

    def test_pgrk(self):
        pg = PageRank.fetchFromCSV('test.csv')
        pg.draw('pic.png')
    
    def test_pgrkAlgo1(self):
        pg = PageRank.fetchFromCSV('test.csv')
        mhat, mhatT = pg.genAdjacentMatrix()
        print(mhat)
        print(mhatT)
        
    def test_pgrkAlgo2(self):
        pg = PageRank.fetchFromCSV('test.csv')
        pg.draw('pic.png')
        res = pg.pageRank(d=0.85)
        print(sum(res))
        print('=' * 70)
        res = pg.pageRank(callerOrCallee='callee', d=0.85)
        print(sum(res))

    def test_matrix(self):
        import numpy as np
        cc = np.array([[1,2],[2,3],[3,4]])
        ee = np.array([[1,0],[0,3]])
        dd = np.array([[1,2],[2,3],[3,4]])
        print(cc * dd)
        print(np.dot(cc,ee)*dd)
        
        matc = np.matrix(cc)
        mate = np.matrix(ee)
        print(matc*mate)
        

if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
