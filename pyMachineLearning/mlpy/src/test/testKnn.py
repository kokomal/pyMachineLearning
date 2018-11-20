# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("..")
from knn import knn
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt


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
        print(knn.createDataSet()) 
        
    def test_knn_easy(self):  
        group, labels = knn.createDataSet()
        input = array([0, 0])
        print(knn.classify0(input, group, labels, 3))

    # # 测试将原始txt数据转换成矩阵并绘制
    def test_readFile_and_draw(self):
        returnMat, classLabelVector = knn.file2matrix('datingTestSet.txt')  
        print(returnMat)
        print(classLabelVector)  
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(returnMat[:, 1], returnMat[:, 2], 15.0 * array(classLabelVector), 15.0 * array(classLabelVector))
        plt.xlabel('Percentage of Time Spent Playing Video Games')
        plt.ylabel('Liters of Ice Cream Consumed Per Week')
        plt.show()

    def test_autonorm(self):
        returnMat, classLabelVector = knn.file2matrix('datingTestSet.txt') 
        norm, range, minBase = knn.autoNorm(returnMat)
        print(norm)  # #已经均一化的数据
        print(range)  # #幅值
        print(minBase)  # #min最小值
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(norm[:, 1], norm[:, 2], 15.0 * array(classLabelVector), 15.0 * array(classLabelVector))
        plt.xlabel('Percentage of Time Spent Playing Video Games')
        plt.ylabel('Liters of Ice Cream Consumed Per Week')
        plt.show()

    def datingClassTest(self):
        hoRatio = 0.10  # hold out 10%
        datingDataMat, datingLabels = knn.file2matrix('datingTestSet2.txt')  # load data setfrom file
        normMat, ranges, minVals = knn.autoNorm(datingDataMat)
        m = normMat.shape[0]
        print("m=%d" % m)
        numTestVecs = int(m * hoRatio)
        print("numTestVecs=%d" % numTestVecs)
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = knn.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
            print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
            if (classifierResult != datingLabels[i]): errorCount += 1.0
        print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
        print(errorCount)

        
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
