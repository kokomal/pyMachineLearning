# coding:utf-8
'''
Created on 2018-12-4
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
    # print("p0Denom=%d, p1Denom=%d" % (p0Denom, p1Denom))
    return p0Vect, p1Vect, pAbusive


# 分类，P(Ci|w) = P(w|Ci)*P(Ci) / P(w)
# P(w)略掉，P(Ci)在本例中是PAb=0.5
# P(w|Ci) = ∏ P(w|Ci) for each i
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # print("P1=%f, while P0=%f" % (p1, p0))
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


# 测试最大频率
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


# RSS的解析,书中的无法访问，采用
# NASA Image of the Day：
# http://www.nasa.gov/rss/dyn/image_of_the_day.rss
# Yahoo Sports - NBA - Houston Rockets News：
# http://sports.yahoo.com/nba/teams/hou/rss.xml
def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # nasa is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)  # houston is class 0
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen); testSet = []  # create test set
    for i in range(10):  # 不够20
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(nasa, houston):
    import operator
    vocabList, p0V, p1V = localWords(nasa, houston)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("*"*50)
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("*"*50)
    for item in sortedNY:
        print(item[0])


if __name__ == '__main__':
    pass
