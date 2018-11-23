# coding:utf-8
'''
Created on 2018年11月19日
Using Python 3.6.3
@author: chenyuanjun
'''
from numpy import *
import operator
from math import log


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照axis==value进行划分，并且删除axis这列的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最优的axis，这里需要注意如果dataset都是均一值，那就没有必要调用此函数
# 输入为矩阵，最后一列为label
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 最后一列为label，因此只取前n-1个axis
    baseEntropy = calcShannonEnt(dataSet)  # 原始的香农信息熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):  # 遍历所有的axis
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # 提纯value
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        # print('FOR CHOICE %d THE INFOGAIN = %f' % (i, infoGain)) # 打印每一种axis的信息熵削减量
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # 返回最优的axis


# 选取最大次数出现的标签,输入为标签列表
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 递归建树，返回str或是map，这是python的强大之处
# 注意这里的labels并不是具体某一个的最终label，而是featureLabel
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 获得标签列表
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 如果标签列表都是一致的，那就不分裂，返回其FINAL UNIFIED TAG
    if len(dataSet[0]) == 1:  # 如果没有feature了，也不分裂（每次分裂都拆一个feature）
        return majorityCnt(classList)  # 返回其FINAL MAJORITY TAG
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优的axis
    bestFeatLabel = labels[bestFeat]  # 类似于axis的属性标签
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)  # 去重获得axis的所有独立的标签
    for value in uniqueVals:  # 注意uniqueVals不一定都是二元的
        subLabels = labels[:]  # 深拷贝label，避免错乱
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 递归建树
    return myTree


# 使用树进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    #print(firstStr)
    #print(featLabels)
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

# 存储序列化
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

# 反序列化
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)