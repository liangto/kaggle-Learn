import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter
from numpy import *
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import Imputer
# from sklearn.model_selection import cross_val_score

# 导入数据
filename="KNN//data.txt"
test_data=pd.read_table(filename,header=None,sep="\t")
lab=['a','b','c','d']
test_data.columns=lab

# 相关性分析
# sns.set_style('whitegrid')
# sns.pairplot(vars=['a','b','c'],data=test_data,hue='d',size=5)
# plt.show()

dataSet=test_data.drop(['d'],axis=1)
labels=test_data['d']
# print(test_data.head())

# 归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges=maxVals-minVals
    normDataset=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataset=dataSet-tile(minVals,(m,1))
    normDataset=normDataset/tile(ranges,(m,1))
    return normDataset,ranges,minVals

def classify0(inX, dataSet, labels, k):
    '''
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    '''
    # # ------------实现 classify0() 方法的第一种方式-
    # # 1. 距离计算
    # dataSetSize = dataSet.shape[0]
    # #距离度量 度量公式为欧氏距离
    # diffMat = tile(inX, (dataSetSize,1))-dataSet
    # sqDiffMat = diffMat**2
    # sqDistances = sqDiffMat.sum(axis=1)
    # distances = sqDistances**0.5    
    # #将距离排序：从小到大
    # sortedDistIndicies = distances.argsort()
    # # 2. 选择距离最小的k个点
    # classCount={}
    # for i in range(k):
    #     cc=sortedDistIndicies[i]
    #     # print(cc)
    #     voteIlabel =labels.ix[cc][0]
    #     # print(voteIlabel)
    #     classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 

    # # 3. 排序并返回出现最多的那个类型
    # # print(classCount)
    # sortedClassCount = sorted(classCount.items(),key = lambda x:x[1], reverse=True)
    # # print(sortedClassCount[0][0])
    # return sortedClassCount[0][0]


    # ------------实现 classify0() 方法的第二种方式-
    # 1. 距离计算
    # inx - dataset 使用了numpy broadcasting，见 https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    # np.sum() 函数的使用见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html
    dist=np.sum((inX-dataSet)**2,axis=1)**0.5
    # 2. k个最近的标签
    # 函数返回的是索引，因此取前k个索引使用[0 : k]
    # 将这k个标签存在列表k_labels中
    k_labels = [labels.ix[index][0] for index in dist.argsort()[0 : k]]
    # 3. 出现次数最多的标签即为最终类别
    # 使用collections.Counter可以统计各个标签的出现次数，most_common返回出现次数最多的标签tuple，例如[('lable1', 2)]，因此[0][0]可以取出标签值
    label = Counter(k_labels).most_common(1)[0][0]
    print(label)
    return label

def datingtest():
    hoRatio=0.1
    normMat, ranges, minVals = autoNorm(dataSet)
    m=normMat.shape[0]
    numtestvecs=int(m*hoRatio)
    print('numTestVecs=', numtestvecs)
    errorcount=0
    for i in range(numtestvecs):
        testdd=normMat.ix[i,:]
        # print(testdd.head())
        traindd=normMat.ix[numtestvecs:m-1,:].reset_index().drop(['index'],axis=1)  
        # print(traindd.head())
        labelsdd=labels.ix[numtestvecs:m-1].reset_index().drop(['index'],axis=1)  
        # print(labelsdd.head())
        classifierResult = classify0(testdd, traindd,labelsdd , 6)
        print("the classifier came back with: %s, the real answer is: %s" % (str(classifierResult), str(labels[i])))
        if (classifierResult != labels[i]): 
            errorcount += 1.0

    print ("the total error rate is: %s" % (str(errorcount / float(numtestvecs))))
    print (str(errorcount))

datingtest()