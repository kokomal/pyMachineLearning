# -*- coding: utf-8 -*-
# 测试回归树算法
'''
Created on 2010-02-06
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from regressionTree import regTrees
import numpy as np


class MyTest(unittest.TestCase):  # 继承unittest.TestCase

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
    
    def test_loadData(self):
        a = regTrees.loadDataSet('ex00.txt')
        print(a)
    
    def test_eye(self):
        testMat = np.mat(np.eye(4))
        print(testMat)
        mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.75)
        print('LARGER', mat0)  # 第一列元素比0.75大的行
        print('SMALLER', mat1)  # 第一列元素比0.75小的行
        
    def test_createTree00(self):
        myDat1 = regTrees.loadDataSet('ex00.txt')   
        myDat1 = np.mat(myDat1)
        regTree = regTrees.createTree(myDat1) 
        print(regTree)

    def test_createTree0(self):
        myDat1 = regTrees.loadDataSet('ex0.txt')   
        myDat1 = np.mat(myDat1)
        regTree = regTrees.createTree(myDat1) 
        print(regTree)

    def test_createTreeEx2(self):
        myDat1 = regTrees.loadDataSet('ex2.txt')   
        myDat1 = np.mat(myDat1)
        regTree = regTrees.createTree(myDat1) 
        print(regTree)
        myDat1 = regTrees.loadDataSet('ex2.txt')   
        myDat1 = np.mat(myDat1)
        regTree = regTrees.createTree(myDat1, ops=(10000,4)) # 手动调参比较敏感
        print(regTree)

if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
    
