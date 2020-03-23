# https://www.cnblogs.com/itdyb/p/5691958.html


'''
2.数据探索分析：主要是对数据进行缺失值分析与异常值的分析。通过发现原始数据中存在票价为空值，票价最小值为0，折扣率最小值为0、总飞行公里数大于0的记录。
'''

def explore(datafile,exploreoutfile):
    """
    进行数据的探索
    @Dylan
    :param data: 原始数据目录
    :return: 探索后的结果
    """
    data=pd.read_csv(datafile,encoding='utf-8')
    explore=data.describe(percentiles=[],include='all').T####包含了对数据的基本描述，percentiles参数是指定计算多少分位数
    explore['null']=len(data)-explore['count'] ##手动计算空值数
    explore=explore[['null','max','min']]####选取其中的重要列
    explore.columns=['空值数','最大值','最小值']
    """describe()函数自动计算的字段包括：count、unique、top、max、min、std、mean。。。。。
    """
    # explore=explore.fillna(0)
    explore.to_excel(exploreoutfile)
    
'''
3.数据预处理：
3.1数据清洗：
（1）丢弃票价为空的记录
（2）丢弃票价为0、平均折扣率不足0、总飞行公里数大于0的距离
'''

def clean_data(datafile,cleanoutfile):
    """
    进行数据清洗，丢弃票价为空记录，丢弃票价为0，折扣不为0且飞行距离大于0的距离
    @Dylan
    :param data:原始数据
    :return:
    """
    data=pd.read_csv(datafile,encoding='utf-8')
 
    data=data[data['SUM_YR_1'].notnull()*data['SUM_YR_2'].notnull()]####票价值非空才会保留
    ####只保留票价非0的，或者平均折扣率与总飞行记录同时为0 的记录
    index1=data['SUM_YR_1']!=0
    index2=data['SUM_YR_2']!=0
    index3=(data['SEG_KM_SUM']==0) & (data['avg_discount']==0)
 
    data=data[index1 | index2 | index3]
    data.to_excel(cleanoutfile)


'''
3.2属性规约
选择与LRFMC模型指标相关的6个属性：FFP_DATE、LOAD_TIME、FLIGHT_COUNT、avg_discount、SEG_KM_SUM、LAST_TO_END。删除不相关的属性。
3.3 数据变换
由于原始数据没有直接给出LRFMC五个指标，需要自己计算，具体的计算方式为：
（1）L=LOAD_TIME-FFP_DATE
（2）R=LAST_TO_END
（3）F=FLIGHT_COUNT
 (4) M=SEG_KM_SUM
（5）C=avg_discount

数据变换的Python代码如下：
'''

def reduction_data(datafile,reoutfile):
    data=pd.read_excel(cleanoutfile,encoding='utf-8')
    data=data[['LOAD_TIME','FFP_DATE','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
    # data['L']=pd.datetime(data['LOAD_TIME'])-pd.datetime(data['FFP_DATE'])
    # data['L']=int(((parse(data['LOAD_TIME'])-parse(data['FFP_ADTE'])).days)/30)
    ####这四行代码费了我3个小时
    d_ffp=pd.to_datetime(data['FFP_DATE'])
    d_load=pd.to_datetime(data['LOAD_TIME'])
    res=d_load-d_ffp
    data['L']=res.map(lambda x:x/np.timedelta64(30*24*60,'m'))
 
    data['R']=data['LAST_TO_END']
    data['F']=data['FLIGHT_COUNT']
    data['M']=data['SEG_KM_SUM']
    data['C']=data['avg_discount']
    data=data[['L','R','F','M','C']]
    data.to_excel(reoutfile)


# 3.4 数据标准化
def zscore_data(datafile,zscorefile):
    data=pd.read_excel(datafile)
    data=(data-data.mean(axis=0))/data.std(axis=0)
    data.columns=['Z'+i for i in data.columns]
 
    data.to_excel(zscorefile,index=False)


'''
4.建立模型
4.1客户聚类
采用kMeans聚类算法对客户数据进行客户分组，聚成5组'''
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import cycle
 
datafile='./tmp/zscore.xls'
k=5
classoutfile='./tmp/class.xls'
resoutfile='./tmp/result.xls'
data=pd.read_excel(datafile)
 
kmodel=KMeans(n_clusters=k,max_iter=1000)
kmodel.fit(data)
 
# print(kmodel.cluster_centers_)
r1=pd.Series(kmodel.labels_).value_counts()
r2=pd.DataFrame(kmodel.cluster_centers_)
r=pd.concat([r2,r1],axis=1)
r.columns=list(data.columns)+['类别数目']
# print(r)
# r.to_excel(classoutfile,index=False)
 
r=pd.concat([data,pd.Series(kmodel.labels_,index=data.index)],axis=1)
r.columns=list(data.columns)+['聚类类别']
# r.to_excel(resoutfile,index=False)

# 自定义绘图函数进行绘制出每个聚类数据的密度图像：
def density_plot(data):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    p=data.plot(kind='kde',linewidth=2,subplots=True,sharex=False)
    [p[i].set_ylabel('密度') for i in range(k)]
    [p[i].set_title('客户群%d' %i) for i in range(k)]
    plt.legend()
    return plt
