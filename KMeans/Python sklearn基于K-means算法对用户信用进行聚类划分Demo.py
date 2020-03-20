# https://blog.csdn.net/qq_41287993/article/details/85041909

# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans

'''
    @Author     :王磊
    @Date       :2018/12/16
    @Description:基于K-means算法对用户信用进行分类划分
'''


class CreditCard:
    def getSourceData(self, path):
        '''
        获取数据源
        :param path:文件路径
        :return: DataFrame
        '''
        return pd.read_csv(path)

    def getFeature(self, data):
        '''
        将特征数据集合降维为1
        :param data: 需要降维的数据数组
        :return: np.array
        '''
        pca = PCA(n_components=1)
        data = pca.fit_transform(data)
        return data

    def getHistoryBehaviorFeature(self, sourceData):
        '''
        获取降维数据获取这一系列特征值的降维特征
        :param sourceData: 源数据集合
        :return: np.array
        '''
        data = sourceData[['瑕疵户', '逾期', '呆账', '强制停卡记录', '退票', '拒往记录']]
        return self.getFeature(data)

    def getEconomicRisksFeature(self, sourceData):
        '''
        获取降维数据获取这一系列特征值的降维特征
        :param sourceData: 源数据集合
        :return: np.array
        '''
        data = sourceData[['借款余额', '个人月收入', '个人月开销', '家庭月收入', '月刷卡额']]
        return self.getFeature(data)

    def getRisksFeature(self, sourceData):
        '''
        获取降维数据获取这一系列特征值的降维特征
        :param sourceData: 源数据集合
        :return: np.array
        '''
        data = sourceData[['职业', '年龄', '住家']]
        return self.getFeature(data)

    def groupPeople(self, predict, user_id):
        '''
        根据预测值与用户卡编号进行分组
        :param predict: 预测结果
        :param user_id: 用户卡编码
        :return: tuple
        '''
        res1 = []
        res2 = []
        res3 = []
        res4 = []
        res5 = []
        for i in range(len(predict)):
            if predict[i] == 0:
                res1.append(user_id[i])
            elif predict[i] == 1:
                res2.append(user_id[i])
            elif predict[i] == 2:
                res3.append(user_id[i])
            elif predict[i] == 3:
                res4.append(user_id[i])
            elif predict[i] == 4:
                res5.append(user_id[i])
        return res1, res2, res3, res4, res5

    def main(self):
        '''
        主函数
        :return:None
        '''
        # 获取源数据
        sourceData = self.getSourceData("../data/credit_card.csv")
        # 用户编号
        user_id = sourceData['信用卡顾客编号']
        # 分别获取历史行为、经济风险情况、收入风险情况降维特征
        behaviorFeature = self.getHistoryBehaviorFeature(sourceData)
        economicRisksFeature = self.getEconomicRisksFeature(sourceData)
        risksFeature = self.getRisksFeature(sourceData)
        # 将三个特征合并
        allFeatures = np.append(behaviorFeature, economicRisksFeature, axis=1)
        allFeatures = np.append(allFeatures, risksFeature, axis=1)
        # 特征工程归一化
        sdScaler = StandardScaler()
        x_data = sdScaler.fit_transform(allFeatures)
        # 初始化算法构造器，设置聚类数为5
        km = KMeans(n_clusters=5)
        # 训练数据获取模型
        model = km.fit_transform(x_data)
        # 持久化模型
        joblib.dump(model, "c:/Users/asus/Desktop/data/python/model/card_km.model")
        # 获取聚类中心点
        center = km.cluster_centers_
        print("聚类的五个中心分别为：")
        for i in range(len(center)):
            print(center[i])
        # 将源数据集进行分类
        predict = km.predict(x_data)
        res1, res2, res3, res4, res5 = self.groupPeople(predict, user_id)
        print("*" * 20 + "以下为源数据集分类部分" + "*" * 20)
        print("第一类：\r\n一共%d人\r\n%s" % (len(res1), str(res1[:])))
        print("第二类：\r\n一共%d人\r\n%s" % (len(res2), str(res2[:])))
        print("第三类：\r\n一共%d人\r\n%s" % (len(res3), str(res3[:])))
        print("第四类：\r\n一共%d人\r\n%s" % (len(res4), str(res4[:])))
        print("第五类：\r\n一共%d人\r\n%s" % (len(res5), str(res5[:])))
        print("*" * 50)


if __name__ == '__main__':
    cc = CreditCard()
    cc.main()

