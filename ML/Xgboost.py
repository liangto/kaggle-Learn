#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##
# File: d:\06-learn\kaggle-Learn\ML\Xgboost.py
# Project: d:\06-learn\kaggle-Learn\ML
# Created Date: Thursday,2018-04-19 1:09:13 pm
# Author: yongliang yang
# 
# Last Modified: Tuesday,2019-08-20 5:28:51 pm
# Modified By: yongliang yang
# 
# *
##


#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

#%%
data = pd.read_csv('ML/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)



#%%

# my_model=XGBRegressor()
# my_model.fit(train_X,train_y,verbose=False)
# predictions=my_model.predict(test_X)
# from sklearn.metrics import mean_absolute_error
# print("Error:" + str(mean_absolute_error(predictions,test_y)))

my_model=XGBRegressor(n_estimators=1000,learning_rate=0.5)
my_model.fit(train_X,train_y,early_stopping_rounds=5
                ,eval_set=[(test_X,test_y)],verbose=False)
predictions=my_model.predict(test_X)

print("Error:" + str(mean_absolute_error(predictions,test_y)))

#%%


my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)
print("Error:" + str(mean_absolute_error(predictions,test_y)))

#%%
train = pd.read_csv('ML/train.csv')
test = pd.read_csv('ML/test.csv')

from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator

#%%
proto = GenericFeatureStatisticsGenerator().ProtoFromDataFrames([{'name': 'test', 'table': test}])

#%%
from IPython.core.display import display, HTML
import base64
protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-dist/facets-jupyter.html" >
        <facets-overview id="elem"></facets-overview>
        <script>
          document.querySelector("#elem").protoInput = "{protostr}";
        </script>"""
html = HTML_TEMPLATE.format(protostr=protostr)
display(HTML(html))

#%%
