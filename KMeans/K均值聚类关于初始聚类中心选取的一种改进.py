##main.py

from centroids import *
from numpy import *
import time
import matplotlib.pyplot as plt
 
## step 1: load data
print ("step 1: load data...")
dataSet = []
fileIn = open('data3.txt')
for line in fileIn.readlines():  #依次读取每行
    lineArr = line.strip().split('\t') #strip去掉每行头尾空白,对于每一行，
    #split('\t')按照制表符切割字符串，得到的结果构成一个数组，数组的每个元素代表一行中的一列。 
    dataSet.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
 
## step 2: clustering...
print ("step 2: clustering...")
#调用mat()函数可以将数组转换为矩阵，然后可以对矩阵进行一些线性代数的操作
dataSet = mat(dataSet)
k = 5
centroids, clusterAssment = kmeans(dataSet, k)
 
## step 3: show the result
print ("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment)


## centroid.py
#初始聚类中心改进算法
from numpy import *
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))#平方，求和，开方
 
# init centroids with random samples 随机初始质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape#求行列数
    centroids = zeros((k, dim))#创建空矩阵，放初始点
    #第一个点
    index = int(random.uniform(0, numSamples))
    centroids[0, :] = dataSet[index, :]
    #第二个点
    A1=mat(zeros((numSamples, 1)))
    for i in range(numSamples):
        distance = euclDistance(centroids[0, :], dataSet[i, :])
        A1[i] = distance
    centroids[1, :]= dataSet[nonzero(A1[:, 0] == max(A1))[0]]
  
    #第三个点及以后，
    #然后再选择距离前两个点的最短距离最大的那个点作为第三个初始类簇的中心点，
    j = 1
    while j<=k-2:
        mi = mat(zeros((numSamples, 1)))
        for i in range(numSamples):
            distance1 = euclDistance(centroids[j-1, :], dataSet[i-1, :])
            distance2 = euclDistance(centroids[j, :], dataSet[i-1, :])
            mi[i-1] = min([distance1,distance2])
        centroids[1+j, :]= dataSet[nonzero(mi[:, 0] == max(mi))[0]]
        j=j+1
    return centroids
 
# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]#行数
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True
 
    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)#调用初始化质心函数
 
    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist  = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])#调用前面的函数
                if distance < minDist:
                    minDist  = distance
                    minIndex = j
                    
            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
 
        ## step 4: update centroids
        for j in range(k):
            #找出每一类的点
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            #clusterAssment[:, 0].A == j测试所有数据的类相同为true不同为false
            #nonzero()[0]把所有为true的位置写出来
            #pointsInCluster = dataSet[nonzero(clusterAssment[:, 0] == j)[0]]  .A的作用目前不清楚，不加也一样
            #求每一类的中心店
            centroids[j, :] = mean(pointsInCluster, axis = 0)
 
    print('Congratulations, cluster complete!')
    return centroids, clusterAssment
 
# show your cluster only available with 3-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim > 3:
        print("Sorry! I can not draw because the dimension of your data is not 3!")
        return 1
 
    mark = ['r', 'g', 'b', 'y', 'm', 'k']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1
    
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    
    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        ax.scatter(dataSet[i, 0], dataSet[i, 1], dataSet[i, 2], c=mark[markIndex], s=10)
   
    mark = ['r', 'b', 'g', 'k', 'm', 'y']
    # draw the centroids
    for i in range(k):
        ax.scatter(centroids[i, 0], centroids[i, 1], dataSet[i, 2], c=mark[3],s=100)
        #plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
        
plt.show()
 
