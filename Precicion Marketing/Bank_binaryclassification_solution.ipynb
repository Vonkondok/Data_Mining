#「二分类算法」提供银行精准营销解决方案 代码存档
# https://www.cnblogs.com/starcrm/p/11806712.html

import mglearn
from numpy import int64
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFECV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
sns.set(style="darkgrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示

# 字段说明
#
# NO    字段名称    数据类型    字段描述
# 1    ID    Int    客户唯一标识
# 2    age    Int    客户年龄
# 3    job    String    客户的职业
# 4    marital    String    婚姻状况
# 5    education    String    受教育水平
# 6    default    String    是否有违约记录
# 7    balance    Int    每年账户的平均余额
# 8    housing    String    是否有住房贷款
# 9    loan    String    是否有个人贷款
# 10    contact    String    与客户联系的沟通方式
# 11    day    Int    最后一次联系的时间（几号）
# 12    month    String    最后一次联系的时间（月份）
# 13    duration    Int    最后一次联系的交流时长
# 14    campaign    Int    在本次活动中，与该客户交流过的次数
# 15    pdays    Int    距离上次活动最后一次联系该客户，过去了多久（999表示没有联系过）
# 16    previous    Int    在本次活动之前，与该客户交流过的次数
# 17    poutcome    String    上一次活动的结果
# 18    y    Int    预测客户是否会订购定期存款业务
from sklearn.tree import DecisionTreeClassifier

data_train = pd.read_csv('train_set.csv')
data_test = pd.read_csv('test_set.csv')
ids_test = data_test['ID']

print(data_train.shape[0])

# data_train['cppv']=data_train['campaign']+data_train['previous']
# data_test['cppv']=data_test['campaign']+data_test['previous']
# data_train.drop(['campaign','previous'], axis=1, inplace=True)
# data_test.drop(['campaign','previous'], axis=1, inplace=True)

# Rela_grouped=data_train.groupby(['cp'])
# Rela_Survival_Rate=(Rela_grouped.sum()/Rela_grouped.count())['y']
# Rela_count=Rela_grouped.count()['y']
#
# ax1=Rela_count.plot(kind='bar',color='g')
# ax2=ax1.twinx()
# ax2.plot(Rela_Survival_Rate.values,color='r')
# ax1.set_xlabel('Relatives')
# ax1.set_ylabel('Number')
# ax2.set_ylabel('Survival Rate')
# plt.title('Survival Rate by Relatives')
# plt.grid(True,linestyle='-',color='0.7')
# plt.show()

# g = sns.FacetGrid(data_train, col='y')
# g.map(plt.hist, 'day', bins=30)
# plt.show()


print("数值处理1：标签指标one-hot编码处理")


data_train.drop(['ID'], axis=1, inplace=True)
data_test.drop(['ID'], axis=1, inplace=True)

dummy = pd.get_dummies(data_train[['month','job','marital','education','default','housing','loan','contact','poutcome']])
dummyTest = pd.get_dummies(data_test[['month','job','marital','education','default','housing','loan','contact','poutcome']])
data_train = pd.concat([dummy, data_train], axis=1)
data_train.drop(['job','marital','education','default','housing','loan','contact','poutcome'], inplace=True, axis=1)
data_test = pd.concat([dummyTest, data_test], axis=1)
data_test.drop(['job','marital','education','default','housing','loan','contact','poutcome'], inplace=True, axis=1)

data_train['day'].replace([30,13,15,4,14,12,18],4,inplace=True)
data_train['day'].replace([5,20,21,11,8,16,2,3],3,inplace=True)
data_train['day'].replace([17,9,6,27,7,22,28],2,inplace=True)
data_train['day'].replace([23,25,26,10,29,19],1,inplace=True)
data_train['day'].replace([1,24,31],0,inplace=True)

data_test['day'].replace([30,13,15,4,14,12,18],4,inplace=True)
data_test['day'].replace([5,20,21,11,8,16,2,3],3,inplace=True)
data_test['day'].replace([17,9,6,27,7,22,28],2,inplace=True)
data_test['day'].replace([23,25,26,10,29,19],1,inplace=True)
data_test['day'].replace([1,24,31],0,inplace=True)


# data_train['month1'] = data_train.month.apply(lambda x: 4 if x in ['may'] else 0)
# data_train['month1'] = data_train.month.apply(lambda x: 3 if x in ['aug','jul','apr'] else 0)
# data_train['month1'] = data_train.month.apply(lambda x: 2 if x in ['jun','feb','nov','oct'] else 0)
# data_train['month1'] = data_train.month.apply(lambda x: 1 if x in ['sep','mar'] else 0)
#
# data_test['month1'] = data_test.month.apply(lambda x: 4 if x in ['may'] else 0)
# data_test['month1'] = data_test.month.apply(lambda x: 3 if x in ['aug','jul','apr'] else 0)
# data_test['month1'] = data_test.month.apply(lambda x: 2 if x in ['jun','feb','nov','oct'] else 0)
# data_test['month1'] = data_test.month.apply(lambda x: 1 if x in ['sep','mar'] else 0)
# #
data_train.drop(['month'], inplace=True, axis=1)
data_test.drop(['month'], inplace=True, axis=1)
# data_train.drop(['day','job_management','marital_single'], axis=1, inplace=True)
# data_test.drop(['day','job_management','marital_single'], axis=1, inplace=True)


# data_train['month'].replace(['may'],4,inplace=True)
# data_train['month'].replace(['aug','jul','apr'],3,inplace=True)
# data_train['month'].replace(['jun','feb','nov','oct'],2,inplace=True)
# data_train['month'].replace(['sep','mar'],1,inplace=True)
# data_train['month'].replace(['jan','dec'],0,inplace=True)

# 多删特征
# data_train.drop(['age','balance','duration','pdays','previous','day','month','job','marital','education','default','housing','loan','contact','poutcome'], inplace=True, axis=1)
# data_test.drop(['age','balance','duration','pdays','previous','day','month','job','marital','education','default','housing','loan','contact','poutcome'], inplace=True, axis=1)

#default、housing、loan都是2分类的指标，删除其中一个即可
# data_train.drop(['default_no','housing_no','loan_no'], inplace=True, axis=1)
# data_test.drop(['default_no','housing_no','loan_no'], inplace=True, axis=1)


################################
########### 数据整理 ###########
################################

data_train['pdays'].replace(-1,9999,inplace=True)
data_test['pdays'].replace(-1,9999,inplace=True)
print("数值处理2：pdays将-1替换为999")
# data_train.drop(['pdays'], inplace=True, axis=1)
# data_test.drop(['pdays'], inplace=True, axis=1)


# g = sns.FacetGrid(data_train, col='y')
# g.map(plt.hist, 'pdays', bins=20)
# plt.show()
# data_train.drop(['pdays'], inplace=True, axis=1)
# data_test.drop(['pdays'], inplace=True, axis=1)

y = data_train['y']
X = data_train[data_train.columns[: -1]]
# # X.info()
# pdays的平均值先前看到是45，而-1距离45很近，距离max值854很远，故还是需要将所有的-1替换为999
#数据预处理：
#数据中pdays=-1表示从未联络过，替换为999



#对方差较大的数据指标进行变换，MinMaxScaler或者StandardScaler
print("数值处理3：数值指标Scaler变换")
scaler = MinMaxScaler()
# numerical = ['age','balance', 'duration', 'pdays', 'previous']
# X[numerical] = scaler.fit_transform(X[numerical])
# data_test[numerical] = scaler.fit_transform(data_test[numerical])
print(data_test.shape)
X = scaler.fit_transform(X)
data_test = scaler.fit_transform(data_test)

# tsvd = TruncatedSVD(n_components=46)
# data_test = tsvd.fit_transform(data_test)
#数据分割，用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.06, random_state=1)
# X_train = tsvd.fit_transform(X_train)
# X_test = tsvd.fit_transform(X_test)
# print(X_train.shape)

#增加二项式特征
# polynomial_interaction = PolynomialFeatures(degree=2,include_bias=False)
# #增加二项式特征，仅仅是交叉特征
# polynomial_interaction = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
# X_train = polynomial_interaction.fit_transform(X_train)
# X_test = polynomial_interaction.fit_transform(X_test)
# data_test = polynomial_interaction.fit_transform(data_test)
# print('after Polynomial:',X_train.shape)
#
# # #保留99%的信息，进行朱成本分析
# pca = PCA(n_components=100,whiten=True)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)
# data_test = pca.fit_transform(data_test)
# print('after PCA:',X_train.shape)

# #卡方分类筛选
# selector = SelectKBest(f_classif,k=300)
# X_train = selector.fit_transform(X_train,y_train)
# X_test = selector.fit_transform(X_test,y_test)
# print('after SelectKBest:',X_train.shape)

# print(X_train['pdays'])

################################
########### 性能计算 ###########
################################


# print('决策树，分数不理想')
# clf = DecisionTreeClassifier(random_state=11)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print(classification_report(y_test, predictions))
# print(cross_val_score(clf,X_train, y_train,scoring='f1'))
# print(cross_val_score(clf,X_test, y_test,scoring='f1'))
# print(clf.score(X_test, y_test))
#
# y_predprob = clf.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
#
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))
#
print('随机森林，0.919203')
clf = RandomForestClassifier(n_estimators=90, random_state=0,oob_score=True,n_jobs=-1)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))
# print(cross_val_score(clf,X_train, y_train,scoring='f1'))
# print(cross_val_score(clf,X_test, y_test,scoring='f1'))
print(clf.score(X_test, y_test))
y_predprob = clf.predict_proba(X_test)
y_predprob = y_predprob[:, 1]
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))

#穷举随机森林的最佳参数,答案：90
# param_test1 ={'n_estimators':range(10,100,5)}
# gsearch1= GridSearchCV(estimator =RandomForestClassifier(min_samples_split=100,
#                                  min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10),
#                        param_grid =param_test1,scoring='roc_auc',cv=5)
# gsearch1.fit(X_train, y_train)
# print(gsearch1.best_params_)
# y_predprob = gsearch1.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
# predictions = gsearch1.predict(X_test)
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))
#
# print('逻辑回归,0.904655,0.915316')
# # print(X_train)
# #clf = Lasso(alpha=0.5)
# clf = LogisticRegression(random_state=0,solver='newton-cg',class_weight='balanced',penalty='l2',n_jobs=-1)
# # solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’).
# clf.fit(X_train, y_train)
# # clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# # print(classification_report(y_test, predictions))
# # print(cross_val_score(clf,X_train, y_train,scoring='f1'))
# # print(cross_val_score(clf,X_test, y_test,scoring='f1'))
# print(clf.score(X_test, y_test))
# y_predprob = clf.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
#
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))
#
# raletion = pd.DataFrame({"columns":list(data_train.columns)[0:-1], "coef":list(clf.coef_.T)})
# print('相关性：',raletion)

# #穷举逻辑回归的最佳参数,答案:
# # best C : LogisticRegression(C=7.742636826811269, class_weight=None, dual=False,
# #                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,
# #                    max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',
# #                    random_state=None, solver='warn', tol=0.0001, verbose=0,
# #                    warm_start=False)
# penalty = ['l1','l2']
# C=np.logspace(0,4,10)
# hyperparameters = dict(C=C,penalty=penalty)
# gridsearch = GridSearchCV(clf,hyperparameters,cv=5,verbose=0)
# best_clf= gridsearch.fit(X_train, y_train)
# print('best C :',best_clf.best_estimator_)
# print(gridsearch.best_params_)
# y_predprob = gridsearch.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
# predictions = gridsearch.predict(X_test)
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))

# print('AdaBoost')
# clf = AdaBoostClassifier(n_estimators=60, random_state=90)
#
# clf.fit(X_train, y_train)
# predictionsByadaBoost = clf.predict(X_test)
# print(classification_report(y_test, predictionsByadaBoost))
# print(cross_val_score(clf,X_train, y_train,scoring='f1'))
# print(cross_val_score(clf,X_test, y_test,scoring='f1'))
# print(clf.score(X_test, y_test))
# pred = clf.predict_proba(X_test)
# dataPred = pd.DataFrame(pred, columns=['pred0', 'pred'])
# dataPred.drop('pred0', axis=1, inplace=True)
# print(dataPred)
#
# y_predprob = clf.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
#
# predictions_train =  clf.predict(X_train)
# y_predprob_train = clf.predict_proba(X_train)
# y_predprob_train = y_predprob_train[:, 1]
#
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictionsByadaBoost))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))
# print("Accuracy y_train : %.4g" % metrics.accuracy_score(y_train, predictions_train))
# print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob_train))
# #
#
#
# # #
# print('神经网络')
# # ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
# # ‘sgd’ refers to stochastic gradient descent.
# # ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
# clf = MLPClassifier(solver='adam', hidden_layer_sizes=(80,80),
#                     random_state=1)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print(clf.score(X_test, y_test))
# y_predprob = clf.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
#
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))
# print('神经网络 end')
# # #导出结果
ID = list(range(25318,36170))
submission = pd.DataFrame(ID)
submission.rename(columns = {0: 'ID'}, inplace = True)
# 将pred_y从array转化成DataFrame
y_predprob_test = clf.predict_proba(data_test)
y_predprob_test = y_predprob_test[:, 1]
y_predprob_DataFrame = pd.DataFrame(y_predprob_test)
submission['pred'] =y_predprob_DataFrame
submission.to_csv('Result.csv', index = False)

#为防止过拟合而减半步长，最大迭代次数加倍
# gbm1 = GradientBoostingClassifier(learning_rate=0.001, n_estimators=10000, max_depth=7, min_samples_leaf=70,
#                                   min_samples_split=1300, subsample=0.8, random_state=10)
# gbm1.fit(X_train, y_train)
#
# y_pred = gbm1.predict(X_test)
# y_predprob = gbm1.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
#
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
# print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))

# print('KNN近邻，分数不理想')
# clf = KNeighborsClassifier(n_neighbors=5)
# clf.fit(X_train,y_train)
# predictions = clf.predict(X_test)
# print(classification_report(y_test, predictions))
# y_predprob = clf.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]

# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))

# print('SVM支持向量机')
# clf = SVC(kernel='rbf',C=1,gamma='auto',probability=True).fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print(classification_report(y_test, predictions))
# y_predprob = clf.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
#
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))

#朴素贝叶斯
# print('朴素贝叶斯')
# clf = GaussianNB()
#
# clf_sigmoid = CalibratedClassifierCV(clf,cv=5)
# clf_sigmoid.fit(X_train,y_train)
# predictions = clf_sigmoid.predict(X_test)
# y_predprob = clf_sigmoid.predict_proba(X_test)
# y_predprob = y_predprob[:, 1]
#
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))

################################
# AdaBoost选为第一次使用的算法,提交数据
################################
# print('AdaBoost')
# adaBoost = AdaBoostClassifier(n_estimators=50, random_state=11)
# adaBoost.fit(X_train, y_train)
#
# age_null = pd.isnull(data_test['age'])
# data_null = data_test[age_null == True]
# # print(data_null)
#
# id = data_test["ID"]
# print(id)
# X_test.drop(['ID'], axis=1, inplace=True)
#
# submission = pd.DataFrame({
#         "ID": id
#     })
#
# submission[['ID']].astype(int)
# # submission[['ID']] = submission[['ID']].astype(int)
# submission.to_csv('submission.csv', index=False)

# data_test.dropna(inplace=True)
# print(np.isnan(data_test).any())
# submission.replace(np.nan, 0, inplace=True)


# predictionsByadaBoost = adaBoost.predict_proba(X_test)
#
# submission = pd.DataFrame({
#         "ID": id,
#         "pred": predictionsByadaBoost
#     })
# submission.to_csv('submission.csv', index=False)
