# https://blog.csdn.net/wsp_1138886114/article/details/80660020

'''从交易数据中提取RFM特征,代码：'''

'''1 导入数据'''
# Retail_Data_Transction.csv  #交易行为数据：customer_id,trans_date,tran_amount
# Retail_Data_Responce.csv    #营销响应结果数据

#查看交易数据表---营销数据表
!head../数据/Retail_Data_Transction.csv
!head../数据/Retail_Data_Responce.csv

#1.1 导入相关包
import pandas as pd
import datetime as dt
import numpy as np

#查看数据
df = pd.read_csv('head../数据/Retail_Data_Transction.csv',parse_dates = ['trans_date'])
df.head()

#1.2 提起R　F　M y原始特征值
#查看交易最早-最晚日期
print(df['trans_date'].min())
print(df['trans_date'].max())

#1.3 设置当前时间
now = dt.datetime(2015,04,01)
#1.4 构造交易时间间隔变量 hist
df['hist'] = now - df['trans_date']
df['hist'].astype('timedelta64[D]')
df['hist'] = df['hist'] / np.timedelta64(1,'D')
df.head()

#假设只分析近两年的数据
#筛选 hist 变量值小于730 天的数据
df = df[df['hist'] < 730]

#1.5 接下来，对用户进行汇总（groupby）,生成R F M 特征变量   agg()分组函数
rfmTable = df.groupby('customer_id').agg(['hist':'min',             #Recency
                                          'customer_id':'count',    #Frequency
                                          'tran_amount':'sum'])     #Monetary
#1.6 对变量重命名
rfmTable.rename(columns = {'hist':'recency'
                           'customer_id':'frequency'
                           'tran_amount':'monetary'},inplace = True)
rfmTable.head()

'''2 对 RFM 进行简单探索性分析'''
rfmTable.describe()

 #客户 RFM 分析
 #对客户进行RFM原始值进行细分（离散化），得到RFM组合值，并与某营销活动的响应结果关联
#2.1 导入1-从交易数据中提取的RFM特征
%run'1-从交易数据中提取的RFM特征.ipynb'

#2.2 R F M 离散化（分成5等分，每等分20%客户）
quantiles = rfmTable.quantile(q = [0.2,0.4,0.6,0.8])
quantiles

quantiles = quantile.to_dict()      #转成字典，便于后续分段时调用
quantiles

'''
    定义RFM 分段函数
    x: 要分段的变量值
    P: 保存rfm分位数的字典的 key 名称，即 recency, frequency,monetary
    d: 保存rfm分位数的字典
'''

#for recency
def Rclass(x,p,d):
    if x <= d[p][0.2]:
        return 5
    elif x <= d[p][0.4]:
        return 4
    elif x <= d[p][0.6]:
        return 3
    elif x <= d[p][0.8]:
        return 2
    else:
        return 1

#for frequency and monetary
def FMclass(x,p,d):
    if x <= d[p][0.2]:
        return 1
    elif x <= d[p][0.4]:
        return 2
    elif x <= d[p][0.6]:
        return 3
    elif x <= d[p][0.8]:
        return 4
    else:
        return 5

#对2.3 R,F,M 变量进行分段，结果保存在 DataFrame 中（rfmSeg）分段后变量名: R_Seg,F_Seg,M_Seg
#对RFM进行分段离散化处理
rfmSeg = rfmTable
rfmSeg['R_Seg'] = rfmSeg['recency'].apply(Rclass,args = ('recency',quantiles))
rfmSeg['F_Seg'] = rfmSeg['frequency'].apply(FMclass,args = ('frequency',quantiles))
rfmSeg['M_Seg'] = rfmSeg['monetary'].apply(FMclass,args = ('monetary',quantiles))

#2.4 R,F,M组合成RFM（直接位数叠加）
rfmSeg['RFMScore'] = rfmSeg.R_Seg.map(str) + F_Seg.map(str) + M_Seg.map(str)
rfmSeg.head()

#2.5 对RFM分数值降序排列
rfmSeg.sort_values(by = ['RFMScore','monetary'],ascending = [False,False]).head()


'''3 R,F,M 与响应率的关系'''
#导入响应数据
response = pd.read_csv('../数据/Retail_Data_Responce.csv')
response.sort_values('customer_id',inplace = True)
response.head()

rfmSeg.reset_index(inplace = True)
rfm.head()

#把 rfmSeg 和 response 按照 customer_id 整合在一张表中
rfmSeg.sort_values('customer_id',inplace = True)
rfm_response = pd.merge(rfmSeg,response,on = 'customer_id')
rfm_response.head()

%matplotlib inline
#Recency VS Response
ax = rfm_response.groupby('R_Seg').agg('response').mean().plot(kind = 'bar',colormap = 'Blues_r')
ax.set_xlabel('R_Seg')
ax.set_ylabel('Proportion of Responders')

#Frequency VS Response
ax = rfm_response.groupby('F_Seg').agg('response').mean().plot(kind = 'bar',colormap = 'Blues_r')
ax.set_xlabel('F_Seg')
ax.set_ylabel('Proportion of Responders')

#Monetary VS Response
ax = rfm_response.groupby('M_Seg').agg('response').mean().plot(kind = 'bar',colormap = 'Blues_r')
ax.set_xlabel('M_Seg')
ax.set_ylabel('Proportion of Responders')


'''4 响应预测模型训练和选择'''
%run'2-客户RFM分析.ipynb'

#4.1 准备数据
#直接使用没有离散化的 recency,frequency,monetary 三个特征作为 X 变量去预测变量 response

# 准备数据集
from sklearn.model_selection import train_test_split
X = rfm_response[['recency','frequency','monetary']]
y = rfm_response['response']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_stats = 1)

#4.2 对比不同的分类算法
    # 逻辑回归
    # 决策树
    # 支持向量机
    # K近邻
    # 朴树贝叶斯
    # 随机森林
    # Adoboost
    # 梯度提升决策树
    # 多层感知器

import pandas as pd
import numpy as np
from sklearn.neighboer import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdoboostClassifier,GridientBoostingClassifier
from sklearn.naive_bayes import GaussionNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import anc

import matplotlib.pyplot as plt
%matplotlib inline

#循环对9个模型进行训练
classifiers = [
    KNeighborsClassifier(n_neighbors = 5),
    LogisticRegression(),
    SVC(kernel = 'rbf',C = 0.01,probability = True)
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdoboostClassifier(),
    GridientBoostingClassifier(),
    GaussionNB(),
    MLPClassifier()]

#计划用 pandas 的 dataframe 存放每一个模型的 accuracy 和 AUC 值，结果保存在 result 中
cols = ['Classifier','Accuracy','AUC']
result = pd.DataFrame(columns = cols)

#对9 个分类模型进行循环处理，依次输出他们的 accuracy 和 AUC 值，结果保存在 result 中
for clf in classifiers:
    clf.fit(X_train,y_train)
    name = clf.__class__.__name__

    print('='*30)
    print(name)

    print('*******Result**********')
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print('Accuracy:[:,2%]'.format(acc))

    y_pred = clf.predict_proba(X_test)
    fpr,ypr,thresholds = roc_curve(y_test,y_pred[:,1])
    auc_value = auc(fpr,tpr)
    print('AUC:{0:0.2f}'。format(auc_value))

    #把当前循环分类器结果保存在 result_clf 的dataframe中
    result_clf = pd.DataFrame([name,acc*100,auc_value],columns = cols)
    #把 result_clf 合并在 result 中
    result = result.append(result_clf)

print('='*30)


#评估选择最好的模型
#查看不同模型的 auc 和 accuracy
result

#选择最好的算法 GBDT 重新训练模型
favorite_clf = GridientBoostingClassifier()
favorite_clf.fit(X_train,y_train)

#用收益曲线来评估模型效果
#计算收益表
def GainTable(y_true_int,y_pred_prob):
    #y_true_int: 真实值，正例1，负例0
    #y_pred_prob:正例预测概率
    data = pd.DataFrame({'y_true':y_true_int,'prob':y_pred_prob})
    data['prob'] = 1-data['prob']
    #计算十分位数
    data['percentile_gp'] = pd.qcut(data['prob'],q = 10,labels = range(10))
    #按十分位分组
    deciles = data.groupby('percentile_gp',sort = True)
    #定义函数计算实例总数
    def total_count(x):return len(x)
    #定义函数计算实正例数
    def pos_count(x):return np.sum(x)
    #定义函数计算实正例率
    def pos_rate(x):return np.sum(x) / float(len(x))
    out = deciles['y_true'].agg([total_count,pos_count,pos_rate])
    #计算负例数
    out['neg_count'] = out['tatol_count'] - out['pos_count']
    #计算累计正例数
    out['pos_cumsum'] = out['pos_count'].columns()
    total_sum = out['pos_cumsum'] / float(total_sum)
    out['percentile'] = (out.index.astype(int) + 1) / 10.0
    return out[['percentile','total_count','pos_count','neg_count','pos_rate','pos_cover_rate']]

#绘制收益曲线
from matplotlib import pyplot as plt
%matplotlib inline
def plotGainChart(GainTable):
    plt.plot(GainTable['percentile'],GainTable['pos_cover_rate'],'g-')
    plt.plot(GainTable['percentile'],GainTable['pos_cover_rate'],'r--')
    plt.legend(['model','random'])
    plt.show()

#对测试集 X_test 进行预测，得到正例的预测概率
y_prob = favarite_clf.predict_proba(X_test)[:,1]        #取标签等于1的概率
#计算收益表
gaintable = GainTable(y_test,y_prob)
gaintable

#绘制收益曲线
plotGainChart(gaintable)




'''5 保存模型'''
    #有两种持久化的方法：
    # 通过pickle.dump() 方法把模型保存为文件
    # 通过pickle.load() 方法把模型读取（加载）为文件

import pickle
with open('response_model.pickle','wb') as fw:
    pickle.dump(favorite_clf,fw)
#调用系统命名 ls 查看当前目录下的文件
!ls




'''二、部署和应用'''

'''1 读取模型文件'''
 #查看当前目录下的文件
!ls

 #response_model.pickle 为保存模型文件
import pickle
with open('response_model.pickle','rb') as fr:
    model = pickle.load(fr)

'''2 预测并保存结果'''
 #2.1 对新数据集提取RFM特征值

import pandas as pd
import datetime as dt
import numpy as np

 #导入新数据
df = .pd.read_csv('../数据/Retail_Data_Transction。csv',parse_dates = ['trans_date'])
df.head(5)

 #设置当前时间 now = 2015-04-01
now = dt.datetime(2015,4,1)
df['hist'] = now - df['trans_date']
df['hist'].astype('timedelta64')
df['hist'] = df['hist'] / np.timedelta64(1,'D')
df.head()

 #2.2 模型预测
使用model对处理好的数据进行预测，预测结果返回0 或1 概率
我们只取第二列的概率值（即1的概率）
prediction = model.predict_proba(rfmTable)[:,1]
prediction = pd.DataFrame(prediction,columns = ['response_proba'])
prediction.head()

 #把预测结果合并到rfmTable中去
prediction = pd.concat([rfmTable,reset_index(drop = False),prediction],axis = 1)
prediction = prediction.set_index('customer_id')    #customer_id作为索引
prediction = prediction.sort_values(by = ['response_proba'],ascending = False)  
            #按照 response_proba排序
prediction.head()

 #2.3 筛选目标客户
 #基于预测结果，我们按照 response_proba 降序排列之后，
  我们可以筛选一定比例的目标客户作为精准营销对象
 #筛选 top20% 的客户作为目标客户
records = len(prediction)           #获取数据集的总行数
rarget_records = int(0.2*records)   #目标客户的records
target_customer = prediction.iloc[:rarget_records,] #筛选 rarget_records 行
target_customer.head()

 #到处 csv 文件
target_customer.to_csv('target_customer.csv')
 #查看导出的 target_customer.csv 文件
!head target_customer.csv
