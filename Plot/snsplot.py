##==========================================
# snsplot.py
# @author yangyongliang
# @description
# @created Sun Feb 11 2018 11:04:53 GMT+0800 (中国标准时间)
# @last-modified Sun Feb 11 2018 11:04:53 GMT+0800 (中国标准时间)
##==========================================

import pandas as pd
pd.set_option('max_columns', None)
import seaborn as sns
from matplotlib import pyplot as plt
import re
import numpy as np

df = pd.read_csv("Plot//CompleteDataset.csv", index_col=0, low_memory=False)

footballers = df.copy()
footballers['Unit'] = df['Value'].str[-1]
footballers['Value(m)'] = np.where(footballers['Unit'] == '0', 0,
                                   footballers['Value'].str[1:-1].replace(
                                       r'[a-zA-Z]', ''))
footballers['Value(m)'] = footballers['Value(m)'].astype(float)
footballers['Value(m)'] = np.where(footballers['Unit'] == 'M',
                                   footballers['Value(m)'],
                                   footballers['Value(m)'] / 1000)

footballers = footballers.assign(
    Value=footballers['Value(m)'],
    Position=footballers['Preferred Positions'].str.split().str[0])

# print(footballers.head())

# df=footballers[footballers['Position'].isin(['ST','GK'])]

# df=footballers

# g=sns.FacetGrid(df,col='Position',col_wrap=5)
# g.map(sns.kdeplot,'Overall')

# df = footballers[footballers['Position'].isin(['ST', 'GK'])]
# df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]
# g = sns.FacetGrid(df, row="Position", col="Club")
# g.map(sns.violinplot, "Overall")

#散点图
# sns.lmplot(x='Value', y='Overall', markers=['o','*','^'],hue='Position',
#            data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],
#            fit_reg=False)

#箱线图
# f = (footballers
#          .loc[footballers['Position'].isin(['ST', 'GK'])]
#          .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
#     )
# f = f[f["Overall"] >= 80]
# f = f[f["Overall"] < 85]
# f['Aggression'] = f['Aggression'].astype(float)

# sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)

#热力图
# f = (
#     footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]
#         .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
#         .dropna()
# ).corr()

# sns.heatmap(f, annot=True)

#折线图
# from pandas.plotting import parallel_coordinates
# f = (
#     footballers.iloc[:, 12:17]
#         .loc[footballers['Position'].isin(['ST', 'GK'])]
#         .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
#         .dropna()
# )
# f['Position'] = footballers['Position']
# f = f.sample(200)
# parallel_coordinates(f, 'Position')

plt.show()
