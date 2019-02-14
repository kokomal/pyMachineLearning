# coding:utf-8
'''
Created on 2019-02-12
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from linearRegression import linearRegression
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def lwlrTest(testArr, xArr, yArr, k=1.0):  # loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = linearRegression.lwlr(testArr[i], xArr, yArr, k)
    return yHat


def scrapePage(inFile,outFile,yr,numPce,origPrc):
    from bs4 import BeautifulSoup
    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
    soup = BeautifulSoup(fr.read())
    i=1
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print ("item #%d did not sell" % i)
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
            print ("%s\t%d\t%s" % (priceStr,newFlag,title))
            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)
    fw.close()
    print('END!!!')

class MyTest(unittest.TestCase):

    def tearDown(self):
        print('***AFTER ONE TEST***')

    def setUp(self):
        print('***BEFORE ONE TEST***')

    @classmethod
    def tearDownClass(self):
         print('FINISHED...')

    @classmethod
    def setUpClass(self):
        print('STARTING...')
    
    def test_1(self):
        linearRegression.hello() 
        
    def test_load(self):
        print(loadDataSet('ex0.txt'))
        
    def test_standRegres(self):
        xArr, yArr = loadDataSet('ex0.txt')
        ws = linearRegression.standRegres(xArr, yArr)
        print(ws)
        xMat = mat(xArr)
        yMat = mat(yArr)
        yHat = xMat * ws
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
        #-----------------------------------------------------------------
        xCopy = xMat.copy()
        xCopy.sort(0)
        yHat = xCopy * ws
        # print(xCopy[:, 1], '---', yHat)
        ax.plot(xCopy[:, 1], yHat)
        plt.show()
        # corrcoef计算皮尔逊相关系数
        print(corrcoef((xMat * ws).T, yMat))  # 这里不要用yHat，因为已经排好序了 

    def test_lwlr(self):
        xArr, yArr = loadDataSet('ex0.txt')
        print(yArr[0])
        print('k=1.0', linearRegression.lwlr(xArr[0], xArr, yArr, 1.0))
        print('k=0.01', linearRegression.lwlr(xArr[0], xArr, yArr, 0.001))  # 可以预测某一个点，哪怕不在已有数据集内，这就是回归的力量
        print('*' * 70)
        yHat = lwlrTest(xArr, xArr, yArr, 0.003)
        xMat = mat(xArr)
        srtInd = xMat[:, 1].argsort(0)
        # print(srtInd) # 排序的arg序号
        xSort = xMat[srtInd][:, 0, :]  # 注意xmat[srtInd]是200*1*2的矩阵，因此需要转置
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xSort[:, 1], yHat[srtInd])
        ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
        plt.show()
        print('k=0.01', linearRegression.lwlr((1.0, 1.085), xArr, yArr, 0.01))  # 奇怪的是如果k=0.001会求不得逆矩阵
        
    def test_abalone(self):
        abX, abY = loadDataSet('abalone.txt')
        yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
        yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
        yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
        print('01error', linearRegression.rssError(abY[0:99], yHat01.T))  # 不同的k输出的rss残差平方和不一样
        print('1error', linearRegression.rssError(abY[0:99], yHat1.T))
        print('10error', linearRegression.rssError(abY[0:99], yHat10.T))
        print('*' * 70)
        print('new data set 01error', linearRegression.rssError(abY[100:199], yHat01.T))  # 换了一个数据集，反而过拟合了
        print('new data set 1error', linearRegression.rssError(abY[100:199], yHat1.T))
        print('new data set 10error', linearRegression.rssError(abY[100:199], yHat10.T))
        print('*' * 70)
        ws = linearRegression.standRegres(abX[0:99], abY[0:99])
        yHat = mat(abX[100:199]) * ws
        print('new data set standErr', linearRegression.rssError(abY[100:199], yHat.T.A))

    def test_ridge(self):
        abX, abY = loadDataSet('abalone.txt')
        ridgeWeights = linearRegression.ridgeTest(abX, abY)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ridgeWeights)
        plt.show()
        
    def test_stagewise(self):
        abX, abY = loadDataSet('abalone.txt')
        wt = linearRegression.stageWise(abX, abY, 0.01, 200)
        print(wt)
        wt = linearRegression.stageWise(abX, abY, 0.001, 5000)
        print(wt)
        xMat = mat(abX)
        yMat = mat(abY).T
        xMat = linearRegression.regularize(xMat)
        yM = mean(yMat, 0)
        yMat -= yM
        weights = linearRegression.standRegres(xMat, yMat.T) # 标准化的权重矩阵
        print(weights.T)
        
    def test_scrape(self):
        filename = 'html/out.txt'
        fr = open(filename,'r+')
        fr.truncate()
        fr.close() # 每次都删除输出文件
        scrapePage('html/8288.html', filename, 2006, 800, 49.99)
        scrapePage('html/10030.html', filename, 2002, 3096, 269.99)
        scrapePage('html/10179.html', filename, 2007, 5195, 499.99)
        scrapePage('html/10181.html', filename, 2007, 3428, 199.99)
        scrapePage('html/10189.html', filename, 2008, 5922, 299.99)
        scrapePage('html/10196.html', filename, 2009, 3263, 249.99)
        print(loadDataSet(filename))
        
    def test_legoLoad(self):
        filename = 'html/out2.txt' # out2文件为完整的数据， 包括年份，片数，新旧，价格这四个属性
        lgX, lgY = loadDataSet(filename)
        print(shape(lgX))
        print(shape(lgY))
        m,n = shape(lgX)
        lgX1 = mat(ones((m, n+1)))
        lgX1[:,1:5] = mat(lgX)
        print(lgX1)
        ws = linearRegression.standRegres(lgX1, lgY)
        print(ws)
        linearRegression.crossValidation(lgX, lgY)
        print("="*80)
        print(linearRegression.ridgeTest(lgX, lgY)) # 通过输出的回归系数，可以进行取舍，避免噪声干扰，例如此案例，就最好用第四和第二个参数进行操作