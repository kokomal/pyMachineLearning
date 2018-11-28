# coding:utf-8
'''
Created on 2018��11��19��
Using Python 3.6.3
@author: chenyuanjun
'''
from numpy import *
import operator
from os import listdir


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'how', 'him'],  # 注意这里的2个how
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: \"%s\" is not in my Vocabulary!" % word)
    return returnVec


# 训练贝叶斯模型，原始版本
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 多少个样本
    numWords = len(trainMatrix[0])  # 词袋的维度
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # P(Ci)
    p0Num = ones(numWords); p1Num = ones(numWords)  # [1,1,1,...,1,1]，避免含0的情况
    p0Denom = 2.0; p1Denom = 2.0  # 防止除0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # trainMatrix[n] = [1,0,1,...,1]
            p1Num += trainMatrix[i]  # 叠加上去[a0,1+n0, ... 1+m0,...]
            p1Denom += sum(trainMatrix[i])  # 归一化的分母
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = (p1Num / p1Denom)
    p0Vect = (p0Num / p0Denom)
    print("p0Denom=%d, p1Denom=%d" % (p0Denom, p1Denom))
    return p0Vect, p1Vect, pAbusive


# 分类，P(Ci|w) = P(w|Ci)*P(Ci) / P(w)
# P(w)略掉，P(Ci)在本例中是PAb=0.5
# P(w|Ci) = ∏ P(w|Ci) for each i
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    print("P1=%f, while P0=%f" % (p1, p0))
    if p1 > p0:
        return 1
    else:
        return 0


# 训练贝叶斯模型LOG
def trainNB0Log(trainMatrix, trainCategory):
    p0V, p1V, pAbusive = trainNB0(trainMatrix, trainCategory)
    p1Vect = log(p1V)  # change to np.log()，改为log是为了防止多个小数相乘之后下溢
    p0Vect = log(p0V)  # change to np.log()
    return p0Vect, p1Vect, pAbusive


# 改进的词袋模型，不再是[0,1,1,0]之类，而是带有频次[0,3,2,1]
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 分词工具
def textParse(bigString):  # 输入为巨型文本，输出为词数组，单个滤掉
    import re
    listOfTokens = re.split(r'\W+', bigString)  # 正则，忽略标点符号和空格
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 头部小写，长度大于2


# 垃圾邮件测试



if __name__ == '__main__':
    pass
