import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

def CreatDataset():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


#计算数据集的香农熵
def CalcChannonEnt(dataSet):
    numDs=len(Dataset)
    labelCounts={}
    for featVec in dataSet:
        currLabel=featVec[-1]
        if currLabel not in labelCounts.keys:
            labelCounts[currLabel]=0
        labelCounts[currLabel]+=1
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numDs
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

#按照指定特征划分数据集
def SplitDataSet(dataSet,index,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[index]==value:
            reducedFeatVec =featVec[:index]
            reducedFeatVec.extend(featVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0]-1)
    baseEntropy=CalcChannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet)/float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print 'infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature