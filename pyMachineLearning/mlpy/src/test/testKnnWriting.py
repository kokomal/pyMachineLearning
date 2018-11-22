# coding:utf-8
'''
Created on 2018-11-19
Using Python 3.6.3
@author: chenyuanjun
'''
import unittest
import sys
sys.path.append("..")
from knn import knn
from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt

# 测试手写数字的KNN算法
class MyTestWriting(unittest.TestCase):  # 继承unittest.TestCase

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
    
    def test_img2vector(self):
        vec = knn.img2vector('testDigits/0_13.txt')
        print("SHAPE OF VEC %d * %d" % vec.shape)
        print(vec[0, 0:31]) 
    
    def test_writingDigits(self):
        knn.handwritingClassTest('testDigits', 'trainDigits')
        
if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
