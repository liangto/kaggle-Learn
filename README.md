# 机器学习-学习

>机器学习建模基础与数据可视化, 参考 [教程链接](https://www.kaggle.com/learn/overview)


>各类算法学习与理解, 参考 [算法练习](https://github.com/apachecn/MachineLearning)




<!-- 机器学习常用脚本 -->


1.识别问题

在这一步先明确这个问题是分类还是回归。通过问题和数据就可以判断出来，数据由 X 和 label 列构成，label 可以一列也可以多列，可以是二进制也可以是实数，当它为二进制时，问题属于分类，当它为实数时，问题属于回归。



2.数据分成两部分，训练集/测试集

分类问题用 StrtifiedKFold
   
    from sklearn.cross_validation import StratifiedKFold

回归问题用 KFold
    
    from sklearn.cross_validation import KFold



3.构造特征-编码

这个时候，需要将数据转化成模型需要的形式。数据有三种类型：数字，类别，文字。当数据是类别的形式时，需要将它的每一类提取出来作为单独一列，然后用二进制表示每条记录相应的值

record 1: 性别 女
record 2：性别 女
record 3：性别 男

转化之后就是：

女 男
record 1: 1 0
record 2：1 0
record 3：0 1

    from sklearn.preprocessing import LabelEncoder
或
    
    from sklearn.preprocessing import OneHotEncoder


4.组合数据

处理完 Feature 之后，就将它们组合到一起。
如果数据是稠密的，就可以用 numpy 的 hstack:

    import numpy as np
    X = np.hstack((x1, x2, ...))

如果是稀疏的，就用 sparse 的 hstack：

    from scipy import sparse
    X = sparse.hstack((x1, x2, ...))
