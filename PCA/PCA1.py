# https://blog.csdn.net/weixin_45435206/article/details/102940365

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
N=200                                                #维度                   

def distance_fun(p1, p2, N):
    result=0
    for i in range(0,N):
        result=result+((p1[i]-p2[i])**2)
    return np.sqrt(result)

def mean_fun(a):
     return np.mean(a,axis=0)

def farthest(center_arr, arr):
    f = [0, 0]
    max_d = 0
    for e in arr:
        d = 0
        for i in range(center_arr.__len__()):
            d = d + np.sqrt(distance_fun(center_arr[i], e, N))
        if d > max_d:
            max_d = d
            f = e
    return f

def closest(a, arr):
    c = arr[1]
    min_d = distance_fun(a, arr[1])
    arr = arr[1:]
    for e in arr:
        d = distance_fun(a, e)
        if d < min_d:
            min_d = d
            c = e
    return c

def pca(XMat):
    average = mean_fun(XMat) 
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #依照featValue进行从大到小排序
    
    sumfeatvalue=sum(index)
    sumt=0
    k=0
    while(sumt<0.9*sumfeatvalue):
        sumt+=index[k]
        k+=1
        
    finalData = []
    selectVec = np.matrix(featVec.T[index]) #所以这里须要进行转置
    finalData = data_adjust * selectVec.T 
    reconData = (finalData * selectVec) + average  
    
    return finalData, reconData,k

def plotBestFit(data1, data2):    
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        axis_x2.append(dataArr2[i,0]) 
        axis_y2.append(dataArr2[i,1])                 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    #ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=1, c='blue')
    plt.show()  
    
if __name__ == "__main__":
    '''
    arr = np.random.randint(0,10000, size=(1000, 1, N))[:, 0, :]
    XMat=arr
    
    '''
    block1=  np.random.randint(0,2000, size=(100000, 1, N))[:, 0, :]     #分区间生成随机数
    block2 = np.random.randint(2000,4000, size=(100000, 1, N))[:, 0, :]
    block3 = np.random.randint(4000,6000, size=(100000, 1, N))[:, 0, :]
    block4 = np.random.randint(6000,8000, size=(100000, 1, N))[:, 0, :]
    block5 = np.random.randint(8000,10000, size=(100000, 1, N))[:, 0, :]
    XMat=np.vstack((block1,block2,block3,block4,block5))
    
    
    
  
    finalData, reconMat,pcaN = pca(XMat) 
    plotBestFit(finalData, reconMat)   #输出前两维切片检查效果
    print('降维到：',pcaN)


