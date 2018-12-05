# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from bayes import bayes
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt
import io


# 测试朴素贝叶斯算法
class TestBayes(unittest.TestCase):  # 继承unittest.TestCase

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
    
    def test_loadDataSet(self):
        print(bayes.loadDataSet()) 

    def test_dataUnique(self):
        print(bayes.createVocabList(bayes.loadDataSet()[0]))  # 获得所有词集合，去重
    
    def test_word2vec(self):
        word = "my dog ate the food on the garbage"
        vocabList = bayes.createVocabList(bayes.loadDataSet()[0])
        print("THE VOCABLIST IS %s" % vocabList)
        print(bayes.setOfWords2Vec(vocabList, word.split()))

    def test_word2vec2(self):
        raw = bayes.loadDataSet()[0]
        vocabList = bayes.createVocabList(raw)
        print("THE VOCABLIST IS %s" % vocabList)
        for wd in raw:
            print("RAW WORD IS %s" % wd)
            print("OUTPUT VEC IS %s" % bayes.setOfWords2Vec(vocabList, wd)) 

    def test_word2vecBag(self):
        raw = bayes.loadDataSet()[0]
        vocabList = bayes.createVocabList(raw)
        print("THE VOCABLIST IS %s" % vocabList)
        for wd in raw:
            print("RAW WORD IS %s" % wd)
            print("OUTPUT VEC BAG IS %s" % bayes.bagOfWords2VecMN(vocabList, wd))  # 注意输出的向量里面how会有2个

    def test_trainNB(self):
        listOPosts, listClasses = bayes.loadDataSet()
        myVocabList = bayes.createVocabList(listOPosts)
        print(myVocabList)  # 获得所有词集合，去重
        trainMat = []
        for postinDoc in listOPosts:
            trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
        print(trainMat)
        p0V, p1V, pAb = bayes.trainNB0(array(trainMat), array(listClasses))
        print(p0V)
        print(p1V)
        print(pAb)
        sm1 = sum(p0V)
        sm2 = sum(p1V)
        print("sm1=%f, sm2=%f" % (sm1, sm2))  # 和不一定为1，因为做了防除0的改造
        print("USING LOG DISP")
        p0V, p1V, pAb = bayes.trainNB0Log(array(trainMat), array(listClasses))
        print(p0V)
        print(p1V)
        print(pAb)

    def test_NB(self):
        listOPosts, listClasses = bayes.loadDataSet()
        myVocabList = bayes.createVocabList(listOPosts)
        trainMat = []
        for postinDoc in listOPosts:
            trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
        p0V, p1V, pAb = bayes.trainNB0Log(array(trainMat), array(listClasses))
        testEntry = ['love', 'my', 'dalmation']
        thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
        print(thisDoc)
        print(testEntry, 'classified as: ', bayes.classifyNB(thisDoc, p0V, p1V, pAb))
        testEntry = ['stupid', 'garbage']
        thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
        print(thisDoc)
        print(testEntry, 'classified as: ', bayes.classifyNB(thisDoc, p0V, p1V, pAb))

    def test_simpleTextParse(self):  # 测试分词，不去重
        raw = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse venenatis velit tempus, ultricies lectus non, suscipit diam. Pellentesque nunc felis, pellentesque a finibus non, mattis at urna. Cras erat mi, tincidunt at sodales nec, consectetur eu turpis. Aliquam ornare auctor massa sed posuere. Praesent lobortis lorem in feugiat lobortis. Pellentesque dapibus semper tristique. Aenean accumsan odio eget est efficitur, at lobortis nibh hendrerit.Nunc non lacus magna. Duis convallis est quam, vel consequat neque ultricies vel. Mauris fringilla malesuada orci. Curabitur et eros finibus, auctor quam at, dignissim nisi. Phasellus eu ligula feugiat, blandit mi et, tempus odio. Aliquam fringilla sed velit eget finibus. Mauris semper est nulla, id convallis quam sodales a. Donec mollis erat eget lacus consequat, quis aliquam diam porttitor. Etiam fringilla tempus lacus at posuere. In hac habitasse platea dictumst. Etiam eu ligula congue, tempor lacus ac, volutpat nisl. Etiam consectetur tortor in ipsum tempor rhoncus. Nam blandit gravida faucibus. Cras in commodo nibh. Duis lacinia bibendum sodales.'
        print(bayes.textParse(raw))
        raw = io.open('email/ham/%d.txt' % 6, encoding="ISO-8859-1").read()
        print(raw)
        print(bayes.textParse(raw))
    
    def spamTest(self):
        docList = []; classList = []; fullText = []
        for i in range(1, 26):
            wordList = bayes.textParse(io.open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = bayes.textParse(io.open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = bayes.createVocabList(docList)  # create vocabulary
        print("VACABULIST IS %s \nwith VACABULIST size = %d" % (vocabList, len(vocabList)))
        trainingSet = range(50); testSet = []  # create test set
        for i in range(10):
            randIndex = int(random.uniform(0, len(trainingSet)))
            print("RANDOM IS %d" % randIndex)
            testSet.append(trainingSet[randIndex])
            del(list(trainingSet)[randIndex]) # 2和3的语法不一样，这里遵从2.7
        trainMat = []; trainClasses = []
        for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
            trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        print(trainMat)
        p0V, p1V, pSpam = bayes.trainNB0(array(trainMat), array(trainClasses)) # 得到训练后的朴素贝叶斯向量P(w|Ci)和P(Ci)
        errorCount = 0
        for docIndex in testSet:  # classify the remaining items
            wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
            if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
                print("classification error", docList[docIndex])
        print('the error rate is: ', float(errorCount) / len(testSet))
        # return vocabList, fullText

    def test_rangeDel(self):
        xx = range(10)
        del(list(xx)[2])
        print(xx)
        del(xx[2]) 
        print(xx)
        print (sys.version)
        
    # 测试最大词频
    def test_freq(self):
        listOPosts, listClasses = bayes.loadDataSet()
        myVocabList = bayes.createVocabList(listOPosts)
        print(myVocabList)
        zz = bayes.calcMostFreq(myVocabList, 'haha my steak is food, my problems is garbage')
        print(zz)
     
    def test_RSS(self):
        import feedparser
        nasa = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
        houston = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
        topWords = bayes.getTopWords(nasa, houston)
        vocabList, pNasa, pHouston = bayes.localWords(nasa, houston)
        # print(vocabList, pNasa, pHouston)
            
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
