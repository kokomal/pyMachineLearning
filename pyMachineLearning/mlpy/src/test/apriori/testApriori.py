# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from apriori import apriori
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt
import votesmart

S = {'A', 'B', 'C'}


def move(D, frm, to):
    print('Moving Disk of Size %d from %s to %s' % (D, frm, to))


def hannoi(D, frm, to):
    if (D == 1):
        move(D, frm, to)
        return
    inner = (list(S - {frm} - {to}))[0]  # 丑陋的取{A,B,C}除frm和to之外的第三者
    hannoi(D - 1, frm, inner)
    move(D, frm, to)
    hannoi(D - 1, inner, to)


def loadBill(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat

    
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
        
    def test_hello(self):
        apriori.hello()
        
    def test_loadAndFrozen(self):
        dt = apriori.loadDataSet()
        c1 = apriori.createC1(dt)
        # print('C1=', c1)
        D = list(map(set, dt))  # 
        # print('D=', D)
        L1, supp = apriori.scanD(D, c1, 0.5)
        # print(L1)
        print('-' * 80)
        zz = apriori.apriori(dt)
        print(zz)
        print('-' * 80)
        print(zz[0][0], '\n', zz[0][1], '\n', zz[0][2])  # 打印各层的集合
        
    def test_conf(self):
        dt = apriori.loadDataSet()
        L, suppData = apriori.apriori(dt, minSupport=0.5)
        print('L=', L)
        print('SUPP=', suppData)
        rules = apriori.generateRules(L, suppData, minConf=0.7)  # 严格一点
        print(rules)
        print('=' * 80)
        rules = apriori.generateRules(L, suppData, minConf=0.5)  # 降低一点
        print(rules)

    def test_hnt(self):
        hannoi(5, 'A', 'C')

    # api key 获取不了，fail
    def test_bill(self):
        votesmart.votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
        bills = votesmart.votesmart.votes.getBillsByStateRecent()
        print(bills)
        
    def test_loadBill(self):
        pass
    
    def test_mushroom(self):
        mushdataset = [line.split() for line in open('mushroom.dat').readlines()]
        L, suppData = apriori.apriori(mushdataset, minSupport=0.3)
        for item in L[1]:
            if item.intersection({'2'}):print(item)
        print('-'*80)
        for item in L[3]:
            if item.intersection({'2'}):print(item)    
