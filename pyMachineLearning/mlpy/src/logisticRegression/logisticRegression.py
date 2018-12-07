# coding:utf-8

import numpy as np
# 逻辑回归的精髓在于，先初始化一个权重向量，然后不断尝试逼近这个真实的向量，直到误差在容忍范围内，或者达到最大迭代次数
# 如此一来，某一条数据的各个维度的向量的权重就已经知悉，那么根据新数据的输入，乘以weight向量，求sigmoid即可清楚此数据为哪一个标签

# deprecated,在inX极小的时候会造成overflow
# def sigmoid(inX):  # sigmoid仅返回(0,1)区间的浮点值
#    return 1.0 / (1 + np.exp(-inX))


# thanks to simon
def sigmoid(value):
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0    
    a = np.exp(-value)
    return 1.0 / (1.0 + a)


def sigmoid2(inX):
    # return 1.0/(1+exp(-inX))
    # 优化20180412
    if inX >= 0:
        return 1.0 / (1.0 + np.exp(-inX))
    else:
        return np.exp(inX) / (1.0 + np.exp(inX))


# 梯度上升法，求最小
def gradAscent(dataMatIn, classLabels):  # 输入1：m*n;输入2：n*1
    dataMatrix = np.mat(dataMatIn)  # m*n
    labelMat = np.mat(classLabels).transpose()  # labelMat 1*n
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1)) # weight n*1
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # dataMatrix*weight = m*n * n*1 = m*1 基本上非0即1
        error = (labelMat - h)  # 如果算出来的h与label有偏差，则多退少补，即如果error为正，则亏了，要补上；反之亦然
        # 当然如果error已经达标，说明收敛，则无需跑完所有循环 error = m*1
        weights = weights + alpha * dataMatrix.transpose() * error  # 更新weight权重 n*m * m*1= n*1
        # 综合加权所有的数据进行对权重矩阵bias的统计，每次的更新都要进行n*m次计算，对于大数据量，不太合算
    return weights


# 随机梯度上升，粗暴但简明
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)  # n维
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights)) # 选择一个数据集的元素
        error = classLabels[i] - h # 选择的数据集有没有误差 error为一个数值
        print(error)
        weights = weights + alpha * error * dataMatrix[i] # 直接叠加error倍的n维数据
    return weights


# 随机梯度上升2,变化如下：
# 1) alpha不再是恒值，而是随迭代进行逐渐衰减的值，但也不是随着迭代加深而单调下降，这有点类似于模拟退火的思想
# 2) 总迭代次数可以控制
# 3) 随机取已有数据，并且不会重复
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha震荡衰减，但不会到0，类似模拟退火
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights)) # h和error仍然是一个数值
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(dataMat, labelMat, weights):
    import matplotlib.pyplot as plt
    #dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    # print(dataArr)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2] # w0*1+w1*x+w2*y=0 --> y=(-w0-w1*x)/w2
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0
    
