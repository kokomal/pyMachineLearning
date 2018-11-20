# coding:utf-8
'''
Created on 2018年11月19日
Using Python 3.6.3
@author: chenyuanjun
'''
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print(diffMat)
    sqDiffMat = diffMat ** 2
    # print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    # print(sqDistances)
    distances = sqDistances ** 0.5  # # 算距离差距矩阵
    classCount = {}
    sortedDistIndicies = distances.argsort()  # # argsort获得各个值的排名, 元数据的越大的数字，其序号排在越前
    # print(distances)
    # print(sortedDistIndicies)
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1  # # 更新排名计数
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] 

    
if __name__ == '__main__':
    group, labels = createDataSet()
    # print(group)
    input = array([0, 0])
    print(classify0(input, group, labels, 3))
