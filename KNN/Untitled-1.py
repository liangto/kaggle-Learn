# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# ## 散点图

#%%
import pandas as pd
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns


#%%
midwest=pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")
categories=np.unique(midwest['category'])
colors=[plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]


#%%
plt.figure(figsize=(16,10),dpi=80,facecolor='w',edgecolor='b')
for i,category in enumerate(categories):
    plt.scatter('area','poptotal',data=midwest.loc[midwest.category==category,:],s=20,c=colors[i],label=str(category))
# plt.gca().set(xlim=(0.0,0.1),ylim=(0,90000),xlabel='Area',ylabel='Population')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("xasd",fontsize=22)
plt.xlabel("Area",fontsize=15)
plt.ylabel('Population',fontsize=15)
plt.xlim(0.0,0.2)
plt.legend(fontsize=12)

plt.show()

#%% [markdown]
# ## 气泡图

#%%
plt.figure(figsize=(16,10),dpi=80,facecolor='w',edgecolor='b')
for i,category in enumerate(categories):
    plt.scatter("area","poptotal",data=midwest.loc[midwest.category==category,:],s='dot_size',c=colors[i],label=str(category),linewidths=0.5)
from scipy.spatial import ConvexHull
from matplotlib import patches
def eeee(x,y,ax=None,**kw):
    if not ax:ax=plt.gca()
    p=np.c_[x,y]
    hull=ConvexHull(p)
    poly=plt.Polygon(p[hull.vertices,:],**kw)
    ax.add_patch(poly)
midwest_selet=midwest.loc[midwest.state=='IN',:]
# eeee(midwest_selet.area,midwest_selet.poptotal,ec='k',fc='gold',alpha=0.3)
eeee(midwest_selet.area,midwest_selet.poptotal,ec='firebrick',fc='none',linewidth=1.5)
plt.title("assddd",fontsize=22)
plt.legend(fontsize=14)


#%%
hull


#%%


