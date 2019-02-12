# coding:utf-8
'''
Created on 2019-02-12
Using Python 3.6.3
@author: chenyuanjun
'''
from numpy import *


def hello():
    print('hello')

    
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部加权线性回归，输入为单条记录1*n，输出是单数值y
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # W为m×m的权重矩阵，给每一个元素一个权重
    for j in range(m):  # next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]  # 差异矩阵
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))  # 仅对角有值
    # print('WEIGHT',weights)
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def rssError(yArr, yHatArr):  # yArr and yHatArr both need to be arrays
    return ((yArr - yHatArr) ** 2).sum()


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def regularize(xMat):  # regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # calc mean then subtract it off
    inVar = var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat


# 前向逐步回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))  # testing code remove
    ws = zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        #print(ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat
