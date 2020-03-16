# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from logisticRegression import logisticRegression
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt

def load():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# 测试逻辑回归
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
    
    def test_knn(self):
        print(logisticRegression.sigmoid(33)) 
        print(logisticRegression.sigmoid(-33)) 
    
    def test_loadData(self):
        dataMat,labelMat = load()
        print(dataMat)
        print(labelMat)
    
    def test_logReg(self):
        dataMat, labelMat = load()
        wt = logisticRegression.gradAscent(dataMat, labelMat)
        print(wt)
    
    def test_logRegStoc0(self):
        dataMat, labelMat = load()
        wt = logisticRegression.stocGradAscent0(array(dataMat), array(labelMat))
        print(wt)    
        
    def test_logRegStoc1(self):
        dataMat, labelMat = load()
        wt = logisticRegression.stocGradAscent1(array(dataMat), array(labelMat))
        print(wt)   
        
    def test_plot(self):
        dataMat,labelMat = load()
        wt = logisticRegression.gradAscent(dataMat, labelMat)
        logisticRegression.plotBestFit(dataMat,labelMat, wt.getA())
        
    def test_plotStoc0(self):
        dataMat,labelMat = load()
        wt = logisticRegression.stocGradAscent0(array(dataMat), array(labelMat))
        logisticRegression.plotBestFit(dataMat,labelMat, wt)
    
    def test_plotStoc1(self):
        dataMat,labelMat = load()
        wt = logisticRegression.stocGradAscent1(array(dataMat), array(labelMat))
        logisticRegression.plotBestFit(dataMat,labelMat, wt)
    
    def test_colic(self):
        import numpy as np
        frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
        trainingSet = []; trainingLabels = []
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
        trainWeights = logisticRegression.stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
        errorCount = 0; numTestVec = 0.0
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(logisticRegression.classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
                errorCount += 1
        errorRate = (float(errorCount)/numTestVec)
        print("the error rate of this test is: %f" % errorRate)
        frTrain.close()
        frTest.close()
        return errorRate
    
    def multiTest(self):
        numTests = 10; errorSum = 0.0
        for k in range(numTests):
            errorSum += self.test_colic()
        print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
    
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
