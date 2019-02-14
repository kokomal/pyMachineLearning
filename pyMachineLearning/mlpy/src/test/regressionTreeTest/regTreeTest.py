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
from tkinter import *

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
        
    # 模型树
    def test_createTreeEx3(self):
        myDat1 = regTrees.loadDataSet('exp2.txt')   
        myDat1 = np.mat(myDat1)
        regTree = regTrees.createTree(myDat1, regTrees.modelLeaf, regTrees.modelErr, (1,10)) 
        print(regTree)
        
    # bike测速
    def test_bike(self):
        trainMat = np.mat(regTrees.loadDataSet('bikeTrain.txt'))    
        testMat = np.mat(regTrees.loadDataSet('bikeTest.txt')) 
        #print(testMat, trainMat)
        myTree = regTrees.createTree(trainMat, ops=(1,20)) # 回归树
        yHat = regTrees.createForeCast(myTree, testMat[:,0])
        r1 = np.corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
        print(r1) # 回归树的相关系数
        print("-"*80)
        myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1,20)) # 模型树
        yHat = regTrees.createForeCast(myTree, testMat[:,0], regTrees.modelTreeEval)
        r2 = np.corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
        print(r2) # 模型树的相关系数
    
    def test_tkinter(self):
        root = Tk()
        mylabel = Label(root, text='hello world')
        mylabel.grid()
        root.mainloop()
    
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
    
