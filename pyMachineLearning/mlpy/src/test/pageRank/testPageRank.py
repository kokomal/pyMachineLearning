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

    def test_draw(self):
        # !-*- coding:utf8-*-
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        G.add_node(1)
        G.add_node(2)
        G.add_nodes_from([3, 4, 5, 6])
        G.add_cycle([1, 2, 3, 4])
        G.add_edge(1, 3)
        G.add_edges_from([(3, 5), (3, 6), (6, 7)])
        nx.draw(G, node_size=300, with_labels=True)
        plt.savefig("youxiangtu.png")
        plt.show()
        
    def test_draw2(self):
        G = nx.random_graphs.barabasi_albert_graph(100, 1)  # 生成一个BA无标度网络G
        nx.draw(G, node_size=300, with_labels=True)
        plt.savefig("ba.png")  # 输出方式1: 将图像存为一个png格式的图片文件
        plt.show()  # 输出方式2: 在窗口中显示这幅图像


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
