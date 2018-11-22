# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("..")
from decisionTree import tree
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt

g_dataSet = [[1, 1, 'yes'],
             [1, 1, 'yes'],
             [1, 0, 'no'],
             [0, 1, 'no'],
             [0, 1, 'no']] 
# 测试手写数字的KNN算法
class MyTestDecisionTree(unittest.TestCase):  # 继承unittest.TestCase

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
    
    def test_calcShannonEnt(self):
        dataSet = g_dataSet
        print('SHANNON ENTROPY IS %f' % tree.calcShannonEnt(dataSet))
        dataSet[0][-1] = 'maybe'  # # this makes it more chaotic
        print('SHANNON ENTROPY IS %f' % tree.calcShannonEnt(dataSet)) 

    def test_splitDataSet(self):
        dataSet = g_dataSet
        print(tree.splitDataSet(dataSet, 0, 0)) # 0号val==0，有2个
        print('-'*20)
        print(tree.splitDataSet(dataSet, 0, 1)) # 0号val==1，有3个
    
    def test_chooseBestSplit(self):
        dataSet = g_dataSet
        print(tree.chooseBestFeatureToSplit(dataSet))
        
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
