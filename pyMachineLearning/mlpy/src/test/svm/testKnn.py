# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from svm import svm
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 测试SVM算法
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
    
    def test_Jrand(self):
        for i in range(100):
            print(svm.selectJrand(8, 10))
        
    def test_clip(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x=[];y=[]
        for i in range(100):
            x.append(i)
            j = svm.clipAlpha(i, 40, 20)
            y.append(j)
            print("CLIP %d between 20 and 40 is %d" % (i, j))
        ax.scatter(x,y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    def test_loadData(self):
        dataMat,labelMat = loadDataSet('testSet.txt') # 数据采用1/-1的归类标签
        print(dataMat)
        print(labelMat)
        
    def test_simpleSMO(self):
        dataMat,labelMat = loadDataSet('testSet.txt') # 数据采用1/-1的归类标签
        b, alphas = svm.smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
        print(b)
        print(alphas[alphas>0])
        for i in range(100):
            if alphas[i] > 0.0: print(dataMat[i], labelMat[i])
        
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
