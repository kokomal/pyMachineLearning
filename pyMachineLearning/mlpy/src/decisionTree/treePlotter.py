# coding:utf-8
'''
Created on 2018-11-23
Using Python 3.6.3
@author: chenyuanjun
'''
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 第一个参数是注释内容，xy为箭头坐标， xytext设置注释内容显示的起始位置，arrowprops为箭头属性
def plotNodeTest(nodeTxt, centerPt, parentPt, nodeType):
    createPlotTest.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotNodeProd(nodeTxt, centerPt, parentPt, nodeType):
    createPlotProd.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlotTest():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlotTest.ax1 = plt.subplot(111, frameon=False)  # ticks for demo puropses
    plotNodeTest('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNodeTest('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def createPlotProd(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    #createPlotProd.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    createPlotProd.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree)) # 全局宽度
    plotTree.totalD = float(getTreeDepth(inTree)) # 全局深度
    print("TOTALW=%d, TOTALD=%d" % (plotTree.totalW, plotTree.totalD))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    print("PLOTTREE.XOFF=%f" % plotTree.xOff)
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
    

# 获得tree的leaf的数目，注意map的第一层永远只有一个key
# {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():  # 从二阶开始要遍历其所有的key
        if type(secondDict[key]).__name__ == 'dict':  # value是不是map？
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1
    return numLeafs


# 获得tree的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


# 绘制中间的字
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlotProd.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 绘制tree
def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # 确定了多少leaf
    depth = getTreeDepth(myTree) # 确定了深度
    firstStr = list(myTree)[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    print("CENT=%f, %f" % (cntrPt[0],cntrPt[1]))
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNodeProd(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            print("PLOTTREE.XOFF=%f" % plotTree.xOff)
            plotNodeProd(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
