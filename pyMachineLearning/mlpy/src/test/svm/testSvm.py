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


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels   


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
        x = [];y = []
        for i in range(100):
            x.append(i)
            j = svm.clipAlpha(i, 40, 20)
            y.append(j)
            print("CLIP %d between 20 and 40 is %d" % (i, j))
        ax.scatter(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    def test_loadData(self):
        dataMat, labelMat = loadDataSet('testSet.txt')  # 数据采用1/-1的归类标签
        print(dataMat)
        print(labelMat)
        
    def test_simpleSMO(self):
        dataMat, labelMat = loadDataSet('testSet.txt')  # 数据采用1/-1的归类标签
        b, alphas = svm.smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
        print(b)
        print(alphas[alphas > 0])
        for i in range(100):
            if alphas[i] > 0.0: print(dataMat[i], labelMat[i])
    
    def test_fullSMOKernel(self):
        dataMat, labelMat = loadDataSet('testSet.txt')  # 数据采用1/-1的归类标签
        b, alphas = svm.smoPKernel(dataMat, labelMat, 0.6, 0.001, 40)
        print("-"*70)
        print(b)
        print(alphas[alphas > 0])

    def test_calcWs(self):
        dataMat, labelMat = loadDataSet('testSet.txt')  # 数据采用1/-1的归类标签
        b, alphas = svm.smoPKernel(dataMat, labelMat, 0.6, 0.001, 40)
        ws = svm.calcWs(alphas, dataMat, labelMat)
        print(ws)
        
    def test_nonZero(self):
        zz = mat([[1, 2, 0],
                  [2, 3, 4],
                  [4, 0, 7]])
        xx = zz[:, 0:2].A
        print(xx)
        vv = nonzero(xx)
        print(vv)
        pp = [1, 2, 3, 4, 0, 9, 8]
        vv = nonzero(pp)
        print(vv)
    
    def test_Rbf(self, k1=1.3):
        dataArr, labelArr = loadDataSet('testSetRBF.txt')
        b, alphas = svm.smoPKernel(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
        datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
        svInd = nonzero(alphas.A > 0)[0]
        sVs = datMat[svInd]  # get matrix of only support vectors
        labelSV = labelMat[svInd];
        print("there are %d Support Vectors" % shape(sVs)[0])
        m, n = shape(datMat)
        errorCount = 0
        for i in range(m):
            kernelEval = svm.kernelTrans(sVs, datMat[i, :], ('rbf', k1))
            predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
            if sign(predict) != sign(labelArr[i]): errorCount += 1
        print("the training error rate is: %f" % (float(errorCount) / m))
        print("-"*70)
        dataArr, labelArr = loadDataSet('testSetRBF2.txt')
        errorCount = 0
        datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
        m, n = shape(datMat)
        for i in range(m):
            kernelEval = svm.kernelTrans(sVs, datMat[i, :], ('rbf', k1))
            predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
            if sign(predict) != sign(labelArr[i]): errorCount += 1    
        print("the test error rate is: %f" % (float(errorCount) / m))  

    def test_loadImg(self):
        a, b = loadImages('../knn/trainDigits')
        print(a, b)

    def test_Digits(self, kTup=('rbf', 10)):
        dataArr, labelArr = loadImages('../knn/trainDigits')
        b, alphas = svm.smoPKernel(dataArr, labelArr, 200, 0.0001, 10000, kTup)
        datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
        svInd = nonzero(alphas.A > 0)[0]
        sVs = datMat[svInd] 
        labelSV = labelMat[svInd];
        print("there are %d Support Vectors" % shape(sVs)[0])
        m, n = shape(datMat)
        errorCount = 0
        for i in range(m):
            kernelEval = svm.kernelTrans(sVs, datMat[i, :], kTup)
            predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
            if sign(predict) != sign(labelArr[i]): errorCount += 1
        print("the training error rate is: %f" % (float(errorCount) / m))
        dataArr, labelArr = loadImages('../knn/testDigits')
        errorCount = 0
        datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
        m, n = shape(datMat)
        for i in range(m):
            kernelEval = svm.kernelTrans(sVs, datMat[i, :], kTup)
            predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
            if sign(predict) != sign(labelArr[i]): errorCount += 1    
        print("the test error rate is: %f" % (float(errorCount) / m)) 

    def test_tens(self):
        import tensorflow as tf
           
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
