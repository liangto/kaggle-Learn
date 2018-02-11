##==========================================
# plotlyplot.py
# @author yangyongliang
# @description 尝试失败 -待后续
# @created Sun Feb 11 2018 15:37:24 GMT+0800 (中国标准时间)
# @last-modified Sun Feb 11 2018 15:37:38 GMT+0800 (中国标准时间)
##==========================================

#%%

import pandas as pd
from matplotlib import pyplot as plt
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

review = pd.read_csv("Plot//winemag-data-130k-v2.csv", index_col=0, low_memory=False)
print(review.head())

#%%
iplot([go.Scatter(x=review.head(1000)['points'],y=review.head(1000)['price'],mode='markers')])
plt.show()


#%%
from IPython.display import YouTubeVideo

YouTubeVideo('wG6rdUURU-w', width=800, height=450)
