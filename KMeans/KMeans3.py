#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuweima
"""
 
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy
import time
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf8')
 
if __name__ == '__main__':
    ## step 1: 加载数据
    print "step 1: load data..."
    dataSet = []
    loss = []
    fileIn = open('path')
    for line in fileIn.readlines():
        lineArr = line.strip('\xef\xbb\xbf')      # '\xef\xbb\xbf'是BOM,标识读入的文件是UTF-8编码，需strip()切掉
        lineArr = lineArr.strip().split('\t')      #注意自己文件中每行数据中是用什么对列数据做分割  建议先用Word 规范一下要打开的文件
        dataSet.append([float(lineArr[0])/1.99759326,(float(lineArr[1])-100)/192230])   #数据规范化【0,1】
    print dataSet
 
    #设定不同k值以运算
    for k in range(2,10):
        clf = KMeans(n_clusters=k) #设定k  ！！！！！！！！！！这里就是调用KMeans算法
        s = clf.fit(dataSet) #加载数据集合
        numSamples = len(dataSet)
        centroids = clf.labels_
        print centroids,type(centroids) #显示中心点
        print clf.inertia_  #显示聚类效果
        mark1 = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        #画出所有样例点 属于同一分类的绘制同样的颜色
        for i in xrange(numSamples):
            #markIndex = int(clusterAssment[i, 0])
            plt.plot(dataSet[i][0], dataSet[i][1], mark1[clf.labels_[i]]) #mark[markIndex])
        mark2 = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 画出质点，用特殊图型
        centroids =  clf.cluster_centers_
        for i in range(k):
            plt.plot(centroids[i][0], centroids[i][1], mark2[i], markersize = 12)
            #print centroids[i, 0], centroids[i, 1]
        plt.show()
        loss.append(clf.inertia_)
    for m in range(8):  #因为k 取值是2-9 （！不包括10） m取值是0-7
        plt.plot(m,loss[m],'bo')
    plt.show()
