# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("../../")
from kmeans import kmeans
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


class MyTest(unittest.TestCase):  # �̳�unittest.TestCase

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
        
    def testrandom(self):
        print(random.rand(5, 1))    
        
    def testLoadData(self):
        dataSet = mat(loadDataSet('testSet.txt'))
        print(dataSet)
        print('MIN-0 IS ', min(dataSet[:, 0]))
        print('MIN-1 IS ', min(dataSet[:, 1]))

    def testMean(self):
        dataSet = mat(loadDataSet('testSet.txt'))
        centroid0 = mean(dataSet, axis=0)
        print(centroid0)
        print('*' * 70)
        ll = centroid0.tolist()
        print(ll)
        print('*' * 70)
        print(ll[0])

    def testCentroid(self):
        dataSet = mat(loadDataSet('testSet.txt'))
        print(dataSet)
        print(kmeans.randCent(dataSet, 4))
        
    def testDist(self):
        dataSet = mat(loadDataSet('testSet.txt'))
        print(kmeans.distEclud(dataSet[0], dataSet[1]))
        
    def testKmeans(self):
        dataSet = mat(loadDataSet('testSet.txt'))
        print(kmeans.kMeans(dataSet, 4))

    def testBikMeans(self):
        dataSet = mat(loadDataSet('testSet.txt'))
        kmeans.biKmeans(dataSet, 3)
        
    # due to web issues
    def failTestGeoGrab(self):
        res = kmeans.geoGrab('1 VA Center', 'Augusta, ME')
        print(res)
    
    def testFinalGeo(self):
        kmeans.clusterClubs(5)
