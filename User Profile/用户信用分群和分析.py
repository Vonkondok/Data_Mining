# 【sklearn】K-Means聚类与PCA降维实践 - 用户信用分群和分析

# https://blog.csdn.net/duanlianvip/article/details/100973811?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

'''
本实验使用电信用户的通信行为数据集，进行用户信用分群和分析。
由于是没有标注的训练样本，使用降维和聚类等无监督方法将用户进行分群，
然后对不同群体数据进行人工分析，确定群体的信用行为特点。

本实验中数据集来自开源的电信用户的通信行为数据集，共30000条数据，7个字段：
入网时间、套餐价格、每月流量、每月话费、每月通话时长、欠费金额、欠费月份数。
'''

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline

# 读取训练数据集
# 读取本地的数据信息部分
X = pd.read_csv('./telecom.csv', encoding='utf-8')
print(X.shape)
X.head()

# 数据预处理
# 数据标准化
from sklearn import preprocessing
 
'''
preprocessing.scale()是按照列进行标准化计算，计算公式为:
(X_train[:,0]-X_train[:,0].mean())/X_train[:,0].std()
(X_train[:,0]-np.mean(X_train[:,0]))/np.std(X_train[:,0])//或者
'''
X_scaled = preprocessing.scale(X)  # scale操作之后的数据零均值，单位方差（方差为1）
X_scaled[0:5]


# 进行PCA数据降维
from sklearn.decomposition import PCA
 
# 生成PCA实例
pca = PCA(n_components=3)  # 把维度降至3维
# 进行PCA降维
X_pca = pca.fit_transform(X_scaled)
# 生成降维后的dataframe
X_pca_frame = pd.DataFrame(X_pca, columns=['pca_1', 'pca_2', 'pca_3'])  # 原始数据由(30000, 7)降维至(30000, 3)
X_pca_frame.head()

# 训练简单模型
from sklearn.cluster import KMeans
 
# KMeans算法实例化，将其设置为K=10
est = KMeans(n_clusters=10)
 
# 作用到降维后的数据上
est.fit(X_pca)

'''
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=10, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)
'''

# 取出聚类后的标签
kmeans_clustering_labels = pd.DataFrame(est.labels_, columns=['cluster'])  # 0-9,一共10个标签
 
# 生成有聚类后的dataframe
X_pca_frame = pd.concat([X_pca_frame, kmeans_clustering_labels], axis=1)
 
X_pca_frame.head()


# 对不同的k值进行计算，筛选出最优的K值
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D图形
from sklearn import metrics
 
# KMeans算法实例化，将其设置为K=range(2, 14)
d = {}
fig_reduced_data = plt.figure(figsize=(12, 12))  #画图之前首先设置figure对象，此函数相当于设置一块自定义大小的画布，使得后面的图形输出在这块规定了大小的画布上，其中参数figsize设置画布大小
for k in range(2, 14):
    est = KMeans(n_clusters=k, random_state=111)
    # 作用到降维后的数据上
    y_pred = est.fit_predict(X_pca)
    # 评估不同k值聚类算法效果
    calinski_harabaz_score = metrics.calinski_harabasz_score(X_pca_frame, y_pred)  # X_pca_frame：表示要聚类的样本数据，一般形如（samples，features）的格式。y_pred：即聚类之后得到的label标签，形如（samples，）的格式
    d.update({k: calinski_harabaz_score})
    print('calinski_harabaz_score with k={0} is {1}'.format(k, calinski_harabaz_score))  # CH score的数值越大越好
    # 生成三维图形，每个样本点的坐标分别是三个主成分的值
    ax = plt.subplot(4, 3, k - 1, projection='3d') #将figure设置的画布大小分成几个部分，表示4(row)x3(colu),即将画布分成4x3，四行三列的12块区域，k-1表示选择图形输出的区域在第k-1块，图形输出区域参数必须在“行x列”范围
    ax.scatter(X_pca_frame.pca_1, X_pca_frame.pca_2, X_pca_frame.pca_3, c=y_pred)  # pca_1、pca_2、pca_3为输入数据，c表示颜色序列
    ax.set_xlabel('pca_1')
    ax.set_ylabel('pca_2')
    ax.set_zlabel('pca_3')

# 绘制不同k值对应的score，找到最优的k值
x = []
y = []
for k, score in d.items():
    x.append(k)
    y.append(score)
 
plt.plot(x, y)
plt.xlabel('k value')
plt.ylabel('calinski_harabaz_score')

X.index = X_pca_frame.index  # 返回：RangeIndex(start=0, stop=30000, step=1)
 
# 合并原数据和三个主成分的数据
X_full = pd.concat([X, X_pca_frame], axis=1)
X_full.head()





# 使用箱形图去除异常点原理（有点类似3sigma原理）
# 按每个聚类分组
grouped = X_full.groupby('cluster')
 
result_data = pd.DataFrame()
# 对分组做循环，分别对每组进行去除异常值处理
for name, group in grouped:
    # 每组去除异常值前的个数
    print('Group:{0}, Samples before:{1}'.format(name, group['pca_1'].count()))
    
    desp = group[['pca_1', 'pca_2', 'pca_3']].describe() # 返回每组的数量、均值、标准差、最小值、最大值等数据
    for att in ['pca_1', 'pca_2', 'pca_3']:
        # 去异常值：箱形图
        lower25 = desp.loc['25%', att]
        upper75 = desp.loc['75%', att]
        IQR = upper75 - lower25
        min_value = lower25 - 1.5 * IQR
        max_value = upper75 + 1.5 * IQR
        # 使用统计中的1.5*IQR法则，删除每个聚类中的噪音和异常点
        group = group[(group[att] > min_value) & (group[att] < max_value)]
    result_data = pd.concat([result_data, group], axis=0)
    # 每组去除异常值后的个数
    print('Group:{0}, Samples after:{1}'.format(name, group['pca_1'].count()))
print('Remain sample:', result_data['pca_1'].count())
'''
        Group:0, Samples before:2628
                     pca_1        pca_2        pca_3
        count  2628.000000  2628.000000  2628.000000
        mean      0.576172    -0.351355     0.845832
        std       0.570799     0.439029     0.685016
        min      -0.385017    -1.401124    -0.435847
        25%       0.136194    -0.608690     0.319778
        50%       0.431924    -0.462852     0.763055
        75%       0.908006    -0.288401     1.308205
        max       2.770590     1.784875     3.463926
        Group:0, Samples after:2240
        Group:1, Samples before:6684
                     pca_1        pca_2        pca_3
        count  6684.000000  6684.000000  6684.000000
        mean     -0.814651    -0.355528     0.507371
        std       0.348469     0.211944     0.399592
        min      -1.623047    -0.856733    -0.088787
        25%      -1.112743    -0.454713     0.194762
        50%      -0.869354    -0.391802     0.406229
        75%      -0.540702    -0.327447     0.715365
        max       0.043140     0.998287     1.954580
        Group:1, Samples after:6009
 '''
 
 # 原始数据降维后的可视化
from mpl_toolkits.mplot3d import Axes3D
 
# 生成三维图形，每个样本点的坐标分别是三个主成分的值
fig_reduced_data = plt.figure()
ax_reduced_data = plt.subplot(111, projection='3d')
ax_reduced_data.scatter(X_pca_frame.pca_1.values, X_pca_frame.pca_2.values, X_pca_frame.pca_3.values)
ax_reduced_data.set_xlabel('Component_1')
ax_reduced_data.set_ylabel('Component_2')
ax_reduced_data.set_zlabel('Component_3')

# 设置每个簇对应的颜色
cluster_2_color = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'cyan', 5: 'black', 6: 'magenta', 7: '#fff0f5',
                   8: '#ffdab9', 9: '#ffa500'}
 
colors_clustered_data = X_pca_frame.cluster.map(cluster_2_color)  # 簇名和颜色映射
fig_reduced_data = plt.figure()
ax_clustered_data = plt.subplot(111, projection='3d')
 
# 聚类算法之后的不同簇数据的映射为不同颜色
ax_clustered_data.scatter(X_pca_frame.pca_1.values, X_pca_frame.pca_2.values, X_pca_frame.pca_3.values,
                          c=colors_clustered_data)
ax_clustered_data.set_xlabel('Component_1')
ax_clustered_data.set_ylabel('Component_2')
ax_clustered_data.set_zlabel('Component_3')

# 筛选后的数据聚类可视化
colors_filtered_data = result_data.cluster.map(cluster_2_color)
fig = plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(result_data.pca_1.values, result_data.pca_2.values, result_data.pca_3.values, c=colors_filtered_data)
ax.set_xlabel('Component_1')
ax.set_ylabel('Component_2')
ax.set_zlabel('Component_3')

# 查看各族中的每月话费情况
monthly_Fare = result_data.groupby('cluster').describe().loc[:, u'每月话费']
monthly_Fare

# mean：均值；std：标准差
monthly_Fare[['mean', 'std']].plot(kind='bar', rot=0, legend=True)  # rot可以控制轴标签的旋转度数。legend是否在图上显示图例

# 查看各族中的入网时间情况
access_time = result_data.groupby('cluster').describe().loc[:, u'入网时间']
access_time

access_time[['mean', 'std']].plot(kind='bar', rot=0, legend=True, title='Access Time')

# 查看各族中的欠费金额情况
arrearage = result_data.groupby('cluster').describe().loc[:, u'欠费金额']
arrearage[['mean', 'std']].plot(kind='bar', rot=0, legend=True, title='Arrearage')

# 综合描述
new_column = ['Access_time', u'套餐价格', u'每月流量', 'Monthly_Fare', u'每月通话时长', 'Arrearage', u'欠费月份数', u'pca_1', u'pca_2',
              u'pca_3', u'cluster']
result_data.columns = new_column
result_data.groupby('cluster')[['Monthly_Fare', 'Access_time', 'Arrearage']].mean().plot(kind='bar')  # 每个簇的Monthly_Fare、Access_time、Arrearag的均值放在一块比较
