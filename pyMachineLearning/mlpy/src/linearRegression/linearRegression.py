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
        # print(ws.T)
        lowestError = inf
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


def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


# 交叉验证
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = []; trainY = []
        testX = []; testY = []
        random.shuffle(list(indexList))
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m * 0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat = mat(yArr).T
    meanX = mean(xMat, 0); varX = var(xMat, 0)
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))
