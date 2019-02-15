# coding:utf-8
'''
Created on 2018-12-4
Using Python 3.6.3
@author: chenyuanjun
'''
from numpy import *
import operator


def hello():
    print('hello')

    
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:  # 构建单[]元素的frozenset
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))  # use frozen set so we
                            # can use it as a key in a dict    


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt: ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:  # 这里进行筛除，合格的才放到retList里面
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):  # creates Ck
    # print("LK=" , Lk, 'k=', k)
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk): 
            L1 = list(Lk[i])[:k - 2]; L2 = list(Lk[j])[:k - 2]
            L1.sort(); L2.sort()  # 精髓在于，排好序后，取前k-2个数据（当前肯定是k-1维，每一个取k-2个，不一样的各1个，那么加起来就是k），如果一样，才进行合体，也就是说，避免了大量不合理的判断和合并
            if L1 == L2:  # if first k-2 elements are equal
                # print('Common Prefix=', L1)
                # print('UNION=', Lk[i] | Lk[j])
                retList.append(Lk[i] | Lk[j])  # set union
    # print('finally retList=', retList)
    return retList


def apriori(dataSet, minSupport=0.15):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)  # scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData # support含全部的重要度，包括落选的


def generateRules(L, supportData, minConf=0.7):  # supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):  # only get the sets with two or more items 只考虑大于1的组合，自己1个组合没有意义
        for freqSet in L[i]:
            print('FREQSET',freqSet)
            H1 = [frozenset([item]) for item in freqSet]
            print('H1',H1)
            print('i--',i)
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: # i=1的情况
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # create new list to return
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # calc confidence supp(P-->H) = (P + H)/(P)
        print('CONF OF conseq', conseq, 'is', conf)
        if conf >= minConf: 
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    print('m=',m, 'freqSet', freqSet)
    if (len(freqSet) > (m + 1)):  # try further merging freqSet比m+1大，说明有merge空间
        print('goon')
        Hmp1 = aprioriGen(H, m + 1)  # create Hm+1 new candidates
        print('HMP', Hmp1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print('After prune HMP', Hmp1)
        if (len(Hmp1) > 1):  # need at least two sets to merge
            print("iter...")
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# a7fa40adec6f4a77178799fae4441030