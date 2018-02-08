import pandas as pd 
from matplotlib import pyplot as plt 
# 数据可视化
reviews=pd.read_csv("Plot//winemag-data_first150k.csv")
# print(reviews.head(3))
#柱状图
# reviews['province'].value_counts().head(10).plot.bar()
# (reviews['province'].value_counts().head(10)/len(reviews)).plot.bar()
# reviews['points'].value_counts().sort_index().plot.bar()

# 折线图
# reviews['points'].value_counts().sort_index().plot.line()

#面积图
# reviews['points'].value_counts().sort_index().plot.area()

#直方图
# reviews[reviews['price'] < 200]['price'].plot.hist()

# 散点图
# reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price', y='points')

#热力图
reviews[reviews['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
plt.show()
