# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from fpGrowth import fpGrowth
from numpy import *
import os


# 测试决策树算法
class MyFPGrowthTree(unittest.TestCase):  # 继承unittest.TestCase

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
    
    def test_treeNode(self):
        rootNode = fpGrowth.treeNode('pyramid', 9, None)
        rootNode.children['eye'] = fpGrowth.treeNode('eye', 13, None)
        rootNode.disp()
        rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix', 3, None)
        rootNode.disp()     
        
    def test_doubleArray(self):
        a = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        print(a)
        print(a[1::2])
        print(a[1:])
        
    def test_buildTree(self):
        simpData = fpGrowth.loadSimpDat()
        print(simpData)
        initSet = fpGrowth.createInitSet(simpData)
        print(initSet)
        myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)
        myFPtree.disp()
        print(myHeaderTab)
        print('-' * 80)
        t1 = fpGrowth.findPrefixPath('x', myHeaderTab['x'][1])
        print(t1)
        t2 = fpGrowth.findPrefixPath('z', myHeaderTab['z'][1])
        print(t2)
        t3 = fpGrowth.findPrefixPath('r', myHeaderTab['r'][1])
        print(t3)
        print('-' * 80)
        freqItems = []
        fpGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)      
        print('FreqItems', freqItems)

    def test_kosarak(self):
        parseDat = [line.split() for line in open('kosarak.dat').readlines()]
        initSet = fpGrowth.createInitSet(parseDat)
        myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 30000)
        myFPtree.disp()
        myFreqList = []
        fpGrowth.mineTree(myFPtree, myHeaderTab, 30000, set([]), myFreqList)
        print(myFreqList)


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
