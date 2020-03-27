
from sklearn.datasets import load_iris          #从datasets加载iris数据     
from sklearn.preprocessing import MinMaxScaler  #从preprocessing加载离差标准化模块
from sklearn.cluster import KMeans              #从cluster加载k均值聚类模块
 
iris=load_iris()                                
iris_data=iris['data']                          #提取数据集的特征
iris_target=iris['target']                      #提取数据集的标签
iris_names=iris['feature_names']                #提取特征名
 
scale=MinMaxScaler().fit(iris_data)             #训练规则
iris_dataScale=scale.transform(iris_data)       #应用规则
kmeans=KMeans(n_clusters=3,random_state=123).fit(iris_dataScale) #构建并训练模型
print('构建K-means模型为：',kmeans)

#（2）聚类后通过可视化查看聚类效果，通过sklearn中的TSNE函数实现可视化

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
 
##使用TSNE进行数据降维,降成两维
tsne = TSNE(n_components=2,init='random',random_state=177).fit(iris_data)
df=pd.DataFrame(tsne.embedding_)       ##将原始数据转换为DataFrame
df['labels'] = kmeans.labels_          ##将聚类结果存储进df数据表
 
##提取不同标签的数据
df1 = df[df['labels']==0]
df2 = df[df['labels']==1] 
df3 = df[df['labels']==2] 
 
## 绘制图形
fig = plt.figure(figsize=(9,6)) ##设定空白画布，并制定大小
##用不同的颜色表示不同数据
plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',
    df3[0],df3[1],'gD')
plt.savefig('../tmp/聚类结果.png') 
plt.show() ##显示图片


#（3）FMI评价法判定K-Means聚类模型（需要真实值）
from sklearn.metrics import fowlkes_mallows_score
for i in range(2,7):
    ##构建并训练模型
    kmeans = KMeans(n_clusters = i,random_state=123).fit(iris_data)
    score = fowlkes_mallows_score(iris_target,kmeans.labels_)
    print('iris数据聚%d类FMI评价分值为：%f' %(i,score))
    
'''
iris数据聚2类FMI评价分值为：0.750473
iris数据聚3类FMI评价分值为：0.820808
iris数据聚4类FMI评价分值为：0.753970
iris数据聚5类FMI评价分值为：0.725483
iris数据聚6类FMI评价分值为：0.600691
'''

#（4）使用轮廓系数法评价（不需要真实值对比）
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
silhouettteScore = []
for i in range(2,15):
    ##构建并训练模型
    kmeans = KMeans(n_clusters = i,random_state=123).fit(iris_data)
    score = silhouette_score(iris_data,kmeans.labels_)
    silhouettteScore.append(score)
plt.figure(figsize=(10,6))
plt.plot(range(2,15),silhouettteScore,linewidth=1.5, linestyle="-")
plt.show()

#（5）使用Calinski-Harabasz指数评价kmeans聚类模型（不需要真实值对比）
from sklearn.metrics import calinski_harabaz_score
for i in range(2,7):
    ##构建并训练模型
    kmeans = KMeans(n_clusters = i,random_state=123).fit(iris_data)
    score = calinski_harabaz_score(iris_data,kmeans.labels_)
    print('iris数据聚%d类calinski_harabaz指数为：%f'%(i,score))
