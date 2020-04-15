# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from decisionTree import tree, treePlotter
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt
g_dataSet = [[1, 1, 'yes'],
             [1, 1, 'yes'],
             [1, 0, 'no'],
             [0, 1, 'no'],
             [0, 1, 'no'],
             [3, 0, 'yes']]
g_labels = ['no surfacing', 'flippers']


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


def genTree():
    dataSet = g_dataSet
    labels = g_labels[:]  # 要小心labels被修改了
    dtree = tree.createTree(dataSet, labels)
    return dtree


# 测试决策树算法
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
    
    def test_listmap(self):
        mp = {'11':22, '33':44}
        ll = list(mp)  # list map 输出所有key的顺序
        print(ll)
        
    def test_calcShannonEnt(self):
        dataSet = g_dataSet
        print('SHANNON ENTROPY IS %f' % tree.calcShannonEnt(dataSet))
        dataSet[0][-1] = 'maybe'  # # this makes it more chaotic
        print('SHANNON ENTROPY IS %f' % tree.calcShannonEnt(dataSet)) 

    def test_splitDataSet(self):
        dataSet = g_dataSet
        print(tree.splitDataSet(dataSet, 0, 0))  # 0号val==0，有2个
        print('-' * 20)
        print(tree.splitDataSet(dataSet, 0, 1))  # 0号val==1，有3个
    
    def test_chooseBestSplit(self):
        dataSet = g_dataSet
        print(tree.chooseBestFeatureToSplit(dataSet))
    
    def test_classCount(self):
        tags = ['a', 'b', 'c', 'd', 'e', 'b', 'd', 'e', 'b']
        print('MAJORITY OF TAGS IS %s' % tree.majorityCnt(tags))
    
    def test_createTree(self):
        dataSet = g_dataSet
        labels = g_labels
        print('THE DECISION TREE IS %s' % tree.createTree(dataSet, labels))     
    
    def test_plotNode(self):
        treePlotter.createPlotTest()

    def test_getNumLeafs(self):
        mp = retrieveTree(0)
        print("THE NUM OF LEAVES IS %d" % treePlotter.getNumLeafs(mp))
        mp = retrieveTree(1)
        print("THE NUM OF LEAVES IS %d" % treePlotter.getNumLeafs(mp))
               
    def test_getMaxDepth(self):
        mp = retrieveTree(0)
        print("THE MAXDEPTH IS %d" % treePlotter.getTreeDepth(mp))
        mp = retrieveTree(1)
        print("THE MAXDEPTH IS %d" % treePlotter.getTreeDepth(mp))
    
    def test_usingTreePlot(self):
        mp = retrieveTree(0)
        treePlotter.createPlotProd(mp)
        mp = retrieveTree(1)
        treePlotter.createPlotProd(mp)
    
    def test_realPlot(self):
        dtree = genTree()
        treePlotter.createPlotProd(dtree)
    
    def test_classify(self):
        dtree = retrieveTree(0)
        labels = g_labels
        print(tree.classify(dtree, labels, [1, 0]))

    def test_classif2(self):
        dtree = genTree()
        print(dtree)
        labels = g_labels[:]
        inputData = [1, 1]
        print("FOR INPUT DATA [%s,%s] FINAL LABEL IS %s" % (inputData[0], inputData[1], tree.classify(dtree, labels, inputData)))

    def test_persist(self):
        dtree = genTree()
        tree.storeTree(dtree, 'dtreeStorage.txt')
    
    def test_deserialize(self):
        dtree = tree.grabTree('dtreeStorage.txt')
        print(dtree)
    
    def test_lense(self):
        fr = open('lenses.txt')
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        print(lenses)
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        cpLabels = lensesLabels[:]
        lensesTree = tree.createTree(lenses, cpLabels)
        print(lensesTree)
        treePlotter.createPlotProd(lensesTree)
        inputData = ['pre', 'hyper', 'no', 'normal']
        cpLabels = lensesLabels[:]
        print(tree.classify(lensesTree, cpLabels, inputData))

            
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
