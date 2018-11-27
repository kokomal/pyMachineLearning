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


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
