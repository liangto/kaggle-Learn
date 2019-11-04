#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##
# File: d:\learn\kaggle-Learn\cctest.py
# Project: d:\learn\kaggle-Learn
# Created Date: Thursday,2018-04-19 3:28:26 pm
# Author: yongliang yang
# 
# Last Modified: Thursday,2018-04-19 4:02:40 pm
# Modified By: yongliang yang
# 
# We're doomed!
##


#%%
import pandas as pd 
testData_path='C:\\Users\\yangyl\\cccc.csv'
testData_data=pd.read_csv(testData_path)
test_melbourne_data=testData_data.dropna(axis=0)
print(testData_data.head())


#%%
y=test_melbourne_data.y_flag
melbourne_predictors= [ 'hismaxdpd', 'drawdownamount', 'tenor', 
                        'currentosprincipal', 'curoverdueday']
x=test_melbourne_data[melbourne_predictors]

print(x.head())

#%%
import seaborn as sns
from matplotlib import pyplot as plt 
sns.pairplot(x, hue="species")
plt.show()


#%%
from sklearn.tree import DecisionTreeRegressor
melbourne_model=DecisionTreeRegressor()
melbourne_model.fit(x,y)
from sklearn.metrics import mean_absolute_error
predited_home_prices=melbourne_model.predict(x)
print(mean_absolute_error(y,predited_home_prices))


#%%
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(x, y,random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))



def get_mae(max_leaf_nodes,predictors_train,predictors_val,targ_train,tar_val):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(predictors_train,targ_train)
    preds_val=model.predict(predictors_val)
    mae=mean_absolute_error(tar_val,preds_val)
    return mae


#%%
print ("#"*50)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae=get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Erro : %d" %(max_leaf_nodes,my_mae))
