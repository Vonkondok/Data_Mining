'''
http://www.pythonheidong.com/blog/article/9785/
1）确定指标，常用指标RFM，结合业务：确定观察窗口为2018.9.21-2019.3.21， 只看有过充值的用户，最近一次充值距当天的天数，最近一次登录距今天的天数，累计充值金额，累计充值次数，观看主播数，关注主播数，送礼主播数，累计送礼金额，样本量 84705
2）数据清洗，查看缺失值，采用拉格朗日插值法补充缺失值；
3）处理异常值，用箱型图查看分布情况，永describe.T和分位数查看，再增加方差，极差等指标，确定异常值范围，索引异常值补充为平均值或用拉格朗日插值法
4）用散点密度图查看各指标分布
5）指标进行相关性分析，剔除掉相关性很高的指标
6）对指标进行Z-标准化
7）用Kmeans进行分类，结合业务常分微小中大超R和免费用户，分5类
8）将分类结果画图
9）将分类结果整理成表格，并给予一定业务建议
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv('user_value.csv',index_col='userid')
print(data)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# 查看异常值和缺失值
print(data.describe(percentiles=[0.1,0.25,0.4,0.6,0.8,0.9,0.99]))
# 方法一：设定异常值范围并等于nan,再统一用拉格朗日插值法插值;结果插出来的值非常大，尚未找到原因，且计算非常缓慢，可能要替换更好的插值方法
# data['recharge_n'][data['recharge_n']>250]=np.nan
# data['recharg_m'][data['recharg_m']>30000]=np.nan
# data['followers'][data['followers']>1000] = np.nan
# data['send_n'][data['send_n']>200] = np.nan
# data['send_m'][data['send_m']>40000] = np.nan
# print(data.describe(percentiles=[0.1,0.25,0.4,0.6,0.8,0.9,0.99]))
#
# # 拉格朗日插值
# def poly_intercolumns(s, n, k=5):
# y = s[list(range(n-k,n))+list(range(n+1,n+k+1))]
# y = y[s.notnull()]
# return lagrange(y.index,list(y))(n)
#
# def add_value(data):
# for j in data.columns:
# for i in range(len(data)):
# if (data[j].isnull())[i]:
# data[j][i] = poly_intercolumns(data[j],i)
# return data
#
# data = add_value(data)
# print(data.info())
# data.to_csv('lagrange_data.csv')
#
# # 查看分布
# data = pd.read_csv('lagrange_data.csv')
# print(data.isnull().any())
# describe = data.describe(percentiles=[0.1,0.25,0.5,0.75,0.99,1]).T
# print(describe.info())
# describe['max-min'] = describe['max'] - describe['min']

# 方法二：先用平均值填充缺失值，再将异常值转变成nan,再用平均值补充
def fillna(data):
for i in data.columns:
mean = data[i].mean()
data[i] = data[i].fillna(mean)
return data

data = fillna(data)
print(data.describe())

# 异常值填充
data['followers'].loc[(data['followers']>1000)|(data['followers']<10)] = data['followers'].mean()
data['send_m'].loc[data['send_m']>40000] = data['send_m'].mean()
data['recharg_m'].loc[data['recharg_m']>30000] = data['recharg_m'].mean()
data['send_n'].loc[data['send_n']>200] = data['send_n'].mean()
# 查看散点图矩阵，没有找着特别的相关性
# sns.pairplot(data=data,vars=['recharge_n', 'recharg_m','followers','send_m', 'send_n'])
# 查看相关系数，发现充值金额与送礼金额，充值天数与登录天数，送礼次数与充值次数有较强的相关性，仅保留充值金额、充值天数、充值次数两项
print(data.corr())
data = data.drop(columns=['log_diff_day', 'send_m', 'send_n'])
print(data.describe())
# # 进行Z标准化
data_z = 1.0*(data-data.mean())/data.std()
print(data_z.tail())
# 用Kmeans建模
'''
1、先对数据进行标准化，用新的变量存储
2、用标准化的数据进行建模，用r1=pd.Series(clf.labels_).value_counts()得到每一类别的数量
3、再用r2=pd.Dataframe(clf.cluster_centers_)得到每一类别的聚类中心
4、用r=pd.concat([r2,r1],axis=1)按索引拼接，得到每一类别的聚类中心和数量
5、用r.columns = list(data.columns)+['类别']
通过桃上述步骤，可得到每一类别的不同变量的中心及数量，但这并不是我们想要的结果
6、将处理好的没有标准化的原始数据与lable拼接，得到每一个具体样本的类别
r = pd.concat([data,pd.Series(clf.labels_,index=data.index)],axis=1)
7、再将变量名拼接，r.columns = list(data.columns)+['类别']
这样最终得到每一个样本的类别，做为最终的画图数据'''
k = 4
clf = KMeans(n_clusters=k,n_jobs=4)
clf.fit(data_z)
print(clf.cluster_centers_)
print(clf.n_clusters)
r1 = pd.Series(clf.labels_).value_counts()
r2 = pd.DataFrame(clf.cluster_centers_)
r = pd.concat([r2,r1],axis=1)
r.columns = list(data.columns) + ['lables_n']
print('r1:', r1)
print('r2:', r2)
print('r:', r)
r = pd.concat([data, pd.Series(clf.labels_,index=data.index)],axis=1)
r.columns = list(data.columns) + ['label']
print(r)

plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制聚类后的概率密度图，要先筛选出类别，再对每一个类别做图，把所有图都放到一起
def density_plot(data,label):
p = data.plot(kind='kde', linewidth=2,subplots=True,sharex=False)
[p[i].set_ylabel('密度') for i in range(len(data.iloc[0]))]
plt.legend(loc='best')
return plt

for i in range(k):
density_plot(data[r['label']==i],i)
plt.show()

'''
②数据层面解读：

第1类：最后1次充值时间在40-90天之间，充值金额和次数接近于0，关注主播在70-80，38%
第2类：最后1次充值时间在0-40天之间，充值金额在0-500之间，充值次数0-10，关注主播数在70-80，56%
第3类：最后1次充值时间在0-25天之间，充值金额在0-3000之间，充值次数0-150，关注主播数200-600，4%
第4类：最后1次充值时间在0-10天之间，充值金额在0-30000之间，充值次数0-500，关注主播数0-200，3.5%

③业务层面解读：
第4类：近期有过充值，且累计充值次数和充值金额均非常高，有较好的付费能力和付费习惯，是我们的VIP重量级发展用户，但是他们的关注主播数偏低，引导他们关注更多主播将能更大程度刺激他们充值，建议：引进一批更为优质的主播，或者专门研究下他们关注主播的特征，进行个性化推荐
第3类：关注大量的主播，说明他们对平台主播比较认可，是重点的发展型用户。他们充值金额偏低，说明他们更偏向理性付费，这批用户更在意性价比，建议多出一些活动，刺激这部分用户付费
第2类：有过少量充值，但是时间已经较为久远，可能已经付费流失，这批用户关注的主播非常少，可能付费后的体验较差导致没有持续性付费，降低首次付费的门槛，提升前几次付费的体验，将是刺激他们持续付费的方法
第1类：基本不充钱，也不关注主播，是低价值用户
'''
