# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt
from pca import pca

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

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
        
    def test_hello(self):
        print('hello')
        
    def test_pca1(self):
        dataMat = loadDataSet('testSet.txt')
        print(dataMat)
        lowDMat, reconMat = pca.pca(dataMat, 1)
        print(shape(lowDMat))
        print(shape(reconMat))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
        ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
        plt.show()
        #---------------------------------------------
        lowDMat, reconMat = pca.pca(dataMat, 2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
        ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
        plt.show()