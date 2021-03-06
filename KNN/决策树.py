import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import log
import DecisionTreePlot as dtPlot
from collections import Counter


def CreatDataset():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


#计算数据集的香农熵
def CalcChannonEnt(dataSet):
    #------第一种方式-----
    # numDs=len(dataSet)
    # labelCounts={}
    # for featVec in dataSet:
    #     currLabel=featVec[-1]
    #     if currLabel not in labelCounts.keys:
    #         labelCounts[currLabel]=0
    #     labelCounts[currLabel]+=1
    # shannonEnt=0
    # for key in labelCounts:
    #     prob=float(labelCounts[key])/numDs
    #     shannonEnt-=prob*log(prob,2)

    #------第二种方式-----
    labelCounts = Counter(data[-1] for data in dataSet)
    probs = [p[1] / len(dataSet) for p in labelCounts.items()]
    shannonEnt = sum([-p * log(p, 2) for p in probs])
    return shannonEnt


    # 计算给定数据集的香农墒的函数
    def calc_shannon_ent(self, data_set):
        # 求list的长度，表示计算参与训练的数据量
        num_entries = len(data_set)
        # 计算分类标签label出现的次数
        label_counts = {}
        # the number of unique elements and their occurance
        for featVec in data_set:
            # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
            current_label = featVec[-1]
            # 为所有可能的分类创建字典，如果当前的健值不存在，则扩展字典并将当前健值加入
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
                label_counts[current_label] += 1
        # 对于label标签的占比，求出label标签的香农墒
        shannon_ent = 0.0
        for key in label_counts:
            # 使所有类标签的发生频率计算类别出现的概率
            prob = float(label_counts[key]) / num_entries
            shannon_ent -= prob * math.log(prob, 2)


#按照指定特征划分数据集
def SplitDataSet(dataSet, index, value):
    #-----第一种方式----
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index + 1:])
            retDataSet.append(reducedFeatVec)

    #------第二种方式-----
    # retDataSet=[data for data in dataSet for i,v in enumerate(data) if i>index and v==value]

    return retDataSet


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    #------第一种方式-----
    # numFeatures=len(dataSet[0])-1
    # baseEntropy=CalcChannonEnt(dataSet)
    # bestInfoGain, bestFeature = 0.0, -1
    # for i in range(numFeatures):
    #     featList=[example[i] for example in dataSet]
    #     uniqueVals = set(featList)
    #     newEntropy = 0.0
    #     for value in uniqueVals:
    #         subDataSet = splitDataSet(dataSet, i, value)
    #         # 计算概率
    #         prob = len(subDataSet)/float(len(dataSet))
    #         # 计算信息熵
    #         newEntropy += prob * calcShannonEnt(subDataSet)
    #     infoGain = baseEntropy - newEntropy
    #     print ('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
    #     if (infoGain > bestInfoGain):
    #         bestInfoGain = infoGain
    #         bestFeature = i


    #-----------第二种方式-----
    baseEntropy=CalcChannonEnt(dataSet)
    bestInfoGain=0
    bestFeature=-1
    print(len(dataSet[0])-1)
    for i in range(len(dataSet[0])-1):
        featureCount=Counter([data[i] for data in dataSet])
        newEntropy=sum(feature[1]/float(len(dataSet))*CalcChannonEnt(SplitDataSet(dataSet,i,feature[0])) for feature in featureCount.items())
        infoGain=baseEntropy-newEntropy
        bestFeature=i
        print('No. {0} feature info gain is {1:.3f}'.format(i, infoGain))
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    print(bestFeature)
    return bestFeature


    # 计算给定数据集的香农墒的函数
    def calc_shannon_ent(self, data_set):
        # 求list的长度，表示计算参与训练的数据量
        num_entries = len(data_set)
        # 计算分类标签label出现的次数
        label_counts = {}
        # the number of unique elements and their occurance
        for featVec in data_set:
            # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
            current_label = featVec[-1]
            # 为所有可能的分类创建字典，如果当前的健值不存在，则扩展字典并将当前健值加入
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
                label_counts[current_label] += 1
        # 对于label标签的占比，求出label标签的香农墒
        shannon_ent = 0.0
        for key in label_counts:
            # 使所有类标签的发生频率计算类别出现的概率
            prob = float(label_counts[key]) / num_entries
            shannon_ent -= prob * math.log(prob, 2)

#选择出现最多的一个结果
def majorityCnt(classList):
    #----第一种方式-----
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key = lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]

    #------第二种方式-----
    label=Counter(classList).most_common(1)[0][0]


#创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        print(classList[0])
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLable = labels[bestFeat]

    myTree = {bestFeatLable: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in featValues:
        subLabels = labels[:]
        myTree[bestFeatLable][value] = createTree(
            SplitDataSet(dataSet, bestFeat, value), subLabels)
        print('myTree', value, myTree)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    # -------------- 第一种方法 start --------------
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
    # -------------- 第一种方法 end --------------

    # -------------- 第二种方法 start --------------
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
    # -------------- 第二种方法 start --------------


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = CreatDataset()
    # print myDat, labels

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print '1---', splitDataSet(myDat, 0, 1)
    # print '0---', splitDataSet(myDat, 0, 0)

    # # 计算最好的信息增益的列
    # print chooseBestFeatureToSplit(myDat)

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # 画图可视化展现
    dtPlot.createPlot(myTree)


if __name__ == "__main__":
    fishTest()



