# coding:utf-8
'''
Created on 2018-12-24
Using Python 3.6.3
@author: chenyuanjun
'''
from numpy import *


# 根据阈值threshVal,来划分输入参数m*n的第dimen号纵列，分别为-1和+1的标签
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray  # 输出m*1


# adaboost的核心，找到最佳的单层决策树
# buildstump的原理为，根据权重分配矩阵D获得最优的分类器的参数，和相应的最小误差以及最优估计
# 树桩的参数很简易，即维度dimen，阈值thresh，以及符号lt还是gt；树桩分类器的功能比较弱，但好处是可以根据某一个特定的dimen
# 进行细腻的分拣，而不是在多个dimen上耦合而迷失
def buildStump(dataArr, classLabels, D):  # D在这里是恒定值,m*1维，代表每一条数据的相应权重
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity
    for i in range(n):  # loop over all dimensions
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max();
        # print(rangeMin, rangeMax)
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                # print(threshVal)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # call stump classify with i, j, lessThan
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # calc total error multiplied by D
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# 完整的adaBoost训练
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # init D to all equal
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        print("*"*40, i + 1, "th Iter", "*"*40)
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        print("D:", D.T)
        print("error=", error)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)  # store Stump Params in Array
        print("classEst: ", classEst.T, " classLabels: ", classLabels)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        print('EXPON=', expon)  # 正确的弱化为-α，错误的加强为+α
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst  # 积分分类
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        print('aggErr', aggErrors)
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0: break
    print('finalD=', D)
    return weakClassArr


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)
