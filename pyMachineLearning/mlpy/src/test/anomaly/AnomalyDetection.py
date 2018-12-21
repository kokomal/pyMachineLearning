# -*- coding: utf-8 -*-
# Author: Bob
# Date:   2016.12.22
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio

'''异常检测主运行程序'''
# 基本原理为：
# 1. 选择训练数据集，获得均值和方差mu和sigma2
# 2. 利用上述的方差和均值，在交叉验证集CV上（里面含有一部分的故障节点），获得CV的多元高斯概率密度集pCV
# 3. 采用F1Score法获取最优epsilon
# 4. 根据测试数据集Test，获得高斯概率密度集pTest，直接筛选epsilon，高于则入围，低于则判定故障


def anomalyDetection_example():
    '''加载并显示数据'''
    data = spio.loadmat('data1.mat')
    # print(data)
    X = data['X']
    # print('raw data: ', X, 'shape=', len(X))
    plt = display_2d_data(X, 'bx')
    plt.title("origin data")
    plt.show()
    '''多元高斯分布函数，并可视化拟合的边界'''
    mu, sigma2 = estimateGaussian(X)  # 参数估计（求均值和方差）这里是根据data[X]进行mu和sigma的计算
    print ('Miu And Sigma2: ', mu, sigma2)
    p = multivariateGaussian(X, mu, sigma2)  # 多元高斯分布函数
    print ('Gaussian', p, 'len=', len(p))
    visualizeFit(X, mu, sigma2)  # 显示图像
    
    '''选择异常点（在交叉验证CV上训练得到最好的epsilon）'''
    Xval = data['Xval']
    yval = data['yval']  # y=1代表异常
    pval = multivariateGaussian(Xval, mu, sigma2)  # 计算CV上的概率密度值
    print('pval Len=', len(pval))
    epsilon, F1 = selectThreshold(yval, pval)  # 选择最优的epsilon临界值
    print(u'在CV上得到的最好的epsilon是：%e' % epsilon)
    print(u'对应的F1Score值为：%f' % F1)
    outliers = np.where(p < epsilon)  # 找到小于临界值的异常点，并作图
    plt.plot(X[outliers, 0], X[outliers, 1], 'o', markeredgecolor='r', markerfacecolor='w', markersize=10.)
    plt = display_2d_data(X, 'bx')
    plt.show()
    
    
# 显示二维数据    
def display_2d_data(X, marker):
    plt.plot(X[:, 0], X[:, 1], marker)
    plt.axis('square')
    return plt


# 参数估计函数（就是求均值和方差）
def estimateGaussian(X):
    m, n = X.shape
    # mu = np.zeros((n, 1))
    # sigma2 = np.zeros((n, 1))
    mu = np.mean(X, axis=0)  # axis=0表示列，每列的均值
    sigma2 = np.var(X, axis=0)  # 求每列的方差
    return mu, sigma2

   
# 多元高斯分布函数    
def multivariateGaussian(X, mu, Sigma2):
    k = len(mu)
    if (Sigma2.shape[0] > 1):
        Sigma2 = np.diag(Sigma2)
    '''多元高斯分布函数'''    
    # print('SIGMA2', Sigma2)
    X = X - mu
    print(X.shape)
    argu = (2 * np.pi) ** (-k / 2) * np.linalg.det(Sigma2) ** (-0.5)
    print('ARGMIU', argu)
    Z = np.dot(X, np.linalg.inv(Sigma2)) 
    print(Z.shape)
    Z = Z * X
    print(Z.shape)
    print(Z)
    # print('ZZZ', Z, 'LEN=', len(Z))
    x = -0.5 * np.sum(Z, axis=1)
    print(x)
    p = argu * np.exp(x)  # axis表示每行
    return p  # 返回的是[],以每行的数据进行高斯法计算

    
# 可视化边界
def visualizeFit(X, mu, sigma2):
    x = np.arange(0, 36, 0.5)  # 0-36，步长0.5
    y = np.arange(0, 36, 0.5)
    X1, X2 = np.meshgrid(x, y)  # 要画等高线，所以meshgird
    Z = multivariateGaussian(np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1))), mu, sigma2)  # 计算对应的高斯分布函数,hstack是横向堆叠
    Z = Z.reshape(X1.shape)  # 调整形状
    plt.plot(X[:, 0], X[:, 1], 'rx')
    print(Z)
    if np.sum(np.isinf(Z).astype(float)) == 0:  # 如果计算的为无穷，就不用画了
        print("contour")
        # plt.contourf(X1,X2,Z,10.**np.arange(-20, 0, 3),linewidth=.5)
        CS = plt.contour(X1, X2, Z, 10.**np.arange(-20, 0, 3), color='black', linewidth=.5)  # 画等高线，Z的值在10.**np.arange(-20, 0, 3)
        # plt.clabel(CS)
    plt.show()


# 选择最优的epsilon，即：使F1Score最大    
def selectThreshold(yval, pval):
    '''初始化所需变量'''
    bestEpsilon = 0.
    bestF1 = 0.
    F1 = 0.
    step = (np.max(pval) - np.min(pval)) / 1000
    '''计算'''
    for epsilon in np.arange(np.min(pval), np.max(pval), step):
        cvPrecision = pval < epsilon
        tp = np.sum((cvPrecision == 1) & (yval == 1).ravel()).astype(float)  # sum求和是int型的，需要转为float
        fp = np.sum((cvPrecision == 1) & (yval == 0).ravel()).astype(float)
        fn = np.sum((cvPrecision == 0) & (yval == 1).ravel()).astype(float)
        precision = tp / (tp + fp)  # 精准度
        recision = tp / (tp + fn)  # 召回率
        F1 = (2 * precision * recision) / (precision + recision)  # F1Score计算公式
        if F1 > bestF1:  # 寻找最优的F1 Score
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1


if __name__ == '__main__':
    anomalyDetection_example()
    
