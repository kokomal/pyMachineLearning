# coding:utf-8
'''
Created on 2018-2-24
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from adaBoost import adaBoost
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt
import io


def loadSimpData():
    datMat = matrix([[ 1. , 2.1],
        [ 2. , 1.1],
        [ 1.3, 1. ],
        [ 1. , 1. ],
        [ 2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


# 测试AdaBoost算法
class TestAdaBoost(unittest.TestCase):  # 继承unittest.TestCase

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
    
    def test_loadDataSet(self):
        datMat, classLabels = loadSimpData()
        D = mat(ones((5, 1)) / 5)
        print(adaBoost.buildStump(datMat, classLabels, D)) 

    def test_adaBoostTrain(self):
        datMat, classLabels = loadSimpData()
        print('Finally the weak classifier is: ', adaBoost.adaBoostTrainDS(datMat, classLabels, 9))
        
    def test_adaBoostClassify(self):
        datMat, classLabels = loadSimpData()
        classifyArr = adaBoost.adaBoostTrainDS(datMat, classLabels, 9)
        print(adaBoost.adaClassify([0, 0], classifyArr))
        print("-"*60)
        print(adaBoost.adaClassify([[5, 5], [0, 0]], classifyArr))

           
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
