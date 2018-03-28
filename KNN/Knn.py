import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import operator
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import Imputer
# from sklearn.model_selection import cross_val_score


filename="KNN//data.txt"
test_data=pd.read_table(filename,header=None,sep="\t")
lables=['a','b','c','d']
test_data.columns=lables
# test_data=test_data.drop(['d'],axis=1)
sns.set_style('whitegrid')
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(test_data['c'],test_data['a'],c=test_data['d'])
sns.pairplot(vars=['a','b','c'],data=test_data,hue='d',size=5)

plt.show()
print(test_data.head())