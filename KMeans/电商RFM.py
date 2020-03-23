'''
聚类通常分为以下步骤：
① 业务提出需求
② 根据业务需求，找到核心的指标。有现成的模型的话（如RFM)，可以直接按模型的指标，如果没有，先罗列出比较重要的指标
③ 从数据库用SQL取出数据
④ 对数据进行清洗，标准化/归一化/正则化
⑤ 聚类，如果是现成的模型，则直接聚类即可，如果是拟定的指标，则对各指标进行相关性验证，剔除掉相关性较高的指标，再聚类
⑥ 根据聚类结果，结合业务场景提供建议
https://blog.csdn.net/cindy407/article/details/92151758
'''

# 项目一：电商用户质量RFM聚类分析

from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 导入并清洗数据
data = pd.read_csv('RFM.csv')
data.user_id = data.user_id.astype('str')
print(data.info())
print(data.describe())
X = data.values[:,1:]

# 数据标准化(z_score)
Model = preprocessing.StandardScaler()
X = Model.fit_transform(X)

# 迭代，选择合适的K
ch_score = []
ss_score = []
inertia = []
for k in range(2,10):
    clf = KMeans(n_clusters=k,max_iter=1000)
    pred = clf.fit_predict(X)
    ch = metrics.calinski_harabaz_score(X,pred)
    ss = metrics.silhouette_score(X,pred)
    ch_score.append(ch)
    ss_score.append(ss)
    inertia.append(clf.inertia_)

# 做图对比
fig = plt.figure()
ax1 = fig.add_subplot(131)
plt.plot(list(range(2,10)),ch_score,label='ch',c='y')
plt.title('CH(calinski_harabaz_score)')
plt.legend()

ax2 = fig.add_subplot(132)
plt.plot(list(range(2,10)),ss_score,label='ss',c='b')
plt.title('轮廓系数')
plt.legend()

ax3 = fig.add_subplot(133)
plt.plot(list(range(2,10)),inertia,label='inertia',c='g')
plt.title('inertia')
plt.legend()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']  # 设置正常显示中文
plt.show()

# 本次采用3个指标综合判定聚类质量，CH，轮廓系数和inertia分数，第1和第3均是越大越好，轮廓系数是越小越好，综合来看，聚为3类效果比较好

# 根据最佳的K值，聚类得到结果
model = KMeans(n_clusters=3,max_iter=1000)
model.fit_predict(X)
labels = pd.Series(model.labels_)
centers = pd.DataFrame(model.cluster_centers_)
result1 = pd.concat([centers,labels.value_counts().sort_index(ascending=True)],axis=1) # 将聚类中心和聚类个数拼接在一起
result1.columns = list(data.columns[1:]) + ['counts']
print(result1)
result = pd.concat([data,labels],axis=1)   # 将原始数据和聚类结果拼接在一起
result.columns = list(data.columns)+['label']  # 修改列名
pd.options.display.max_columns = None  # 设定展示所有的列
print(result.groupby(['label']).agg('mean')) # 分组计算各指标的均值

# 对聚类结果做图

fig = plt.figure()
ax1= fig.add_subplot(131)
ax1.plot(list(range(1,4)),s.R_days,c='y',label='R')
plt.title('R指标')
plt.legend()
ax2= fig.add_subplot(132)
ax2.plot(list(range(1,4)),s.F_times,c='b',label='F')
plt.title('F指标')
plt.legend()
ax3= fig.add_subplot(133)
ax3.plot(list(range(1,4)),s.M_money,c='g',label='M')
plt.title('M指标')
plt.legend()
plt.show()
