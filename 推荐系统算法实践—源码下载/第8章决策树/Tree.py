
# coding: utf-8

# ## 1）环境设定

# In[1]:

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from os import path, listdir
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib  
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ## 2）数据准备

# In[2]:

# 数据处理，读取libSVM格式数据，并且将数据归一化，样本划分，并且根据batch参数生成batch
def process_data(data_path, feature_size, test_rat=0.3, random_seed=0, train_batch_size=5000, test_batch_size=5000):
    # 读取文件
    # batch生成
    return {"train_batch": train_batch, "test_batch": test_batch, "X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}

# In[3]:

data_path = '/data/data01/'
test_rat=0.4
random_seed=0
train_batch_size=20000
test_batch_size=20000
feature_size=530

# 获取样本数据
data = process_data(data_path, feature_size, test_rat, random_seed, train_batch_size, test_batch_size)

train_batch = data['train_batch']
test_batch = data['test_batch']

X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape


# ## 3）Tree模型

# In[4]:

# 3.1 随机森林模型，并且设定参数  
rf_model= RandomForestClassifier(
n_estimators=30, 
criterion='gini',  
max_depth=20,
min_samples_leaf=200)

# 3.1 GBDT模型，并且设定参数  
gbdt_model= GradientBoostingClassifier(
n_estimators=30, 
criterion='friedman_mse',  
max_depth=20,
min_samples_leaf=200)

# 3.2 训练模型
rf_model.fit(X_train,Y_train.values.ravel())
gbdt_model.fit(X_train,Y_train.values.ravel())


# In[5]:

# 3.3 采用测试集验证模型离线指标  
# 训练集AUC
probs_train= rf_model.predict_proba(X_train)  
AUC1 = metrics.roc_auc_score(Y_train, probs_train[:,1])
print("RF Train Auc: %s"%(AUC1))

# 测试集AUC
probs_test= rf_model.predict_proba(X_test)  
predict_test = rf_model.predict(X_test)
AUC2 = metrics.roc_auc_score(Y_test, probs_test[:,1])
print("RF Test Auc: %s"%(AUC2))

# 训练集AUC
probs_train2= gbdt_model.predict_proba(X_train)  
AUC3 = metrics.roc_auc_score(Y_train, probs_train2[:,1])
print("Gbdt Train Auc: %s"%(AUC3))

# 测试集AUC
probs_test2= gbdt_model.predict_proba(X_test)  
AUC4 = metrics.roc_auc_score(Y_test, probs_test2[:,1])
print("Gbdt Test Auc: %s"%(AUC4))


# In[6]:

# 准确率
accuracy = metrics.accuracy_score(Y_test, predict_test) 
print("Test Accuracy: %s"%(accuracy))

# 召回率
recall = metrics.recall_score(Y_test, predict_test) 
print("Test Recall: %s"%(recall))

# F1值
f1 = metrics.f1_score(Y_test, predict_test) 
print("Test F1: %s"%(f1))


# In[7]:

# 3.1 随机森林模型，并且设定参数  
rf_model= RandomForestClassifier(
n_estimators=50, 
criterion='gini',  
max_depth=30,
min_samples_leaf=100)

# 3.1 GBDT模型，并且设定参数  
gbdt_model= GradientBoostingClassifier(
n_estimators=50, 
criterion='friedman_mse',  
max_depth=30,
min_samples_leaf=100)

# 3.2 训练模型
rf_model.fit(X_train,Y_train)
gbdt_model.fit(X_train,Y_train)

# 3.3 采用测试集验证模型离线指标  
# RF训练集AUC
probs_train= rf_model.predict_proba(X_train)  
AUC1 = metrics.roc_auc_score(Y_train, probs_train[:,1])
print("RF Train Auc: %s"%(AUC1))

# RF测试集AUC
probs_test= rf_model.predict_proba(X_test)  
predict_test = rf_model.predict(X_test)
AUC2 = metrics.roc_auc_score(Y_test, probs_test[:,1])
print("RF Test Auc: %s"%(AUC2))

# Gbdt训练集AUC
probs_train2= gbdt_model.predict_proba(X_train)  
AUC3 = metrics.roc_auc_score(Y_train, probs_train2[:,1])
print("Gbdt Train Auc: %s"%(AUC3))

# Gbdt测试集AUC
probs_test2= gbdt_model.predict_proba(X_test)  
AUC4 = metrics.roc_auc_score(Y_test, probs_test2[:,1])
print("Gbdt Test Auc: %s"%(AUC4))

# 准确率
accuracy = metrics.accuracy_score(Y_test, predict_test) 
print("Test Accuracy: %s"%(accuracy))

# 召回率
recall = metrics.recall_score(Y_test, predict_test) 
print("Test Recall: %s"%(recall))

# F1值
f1 = metrics.f1_score(Y_test, predict_test) 
print("Test F1: %s"%(f1))

# 3.5 模型保存
joblib.dump(rf_model,"rf_model.model")
joblib.dump(gbdt_model,"gbdt_model.model")  
#模型加载
load_rf = joblib.load("rf_model.model")
load_gbdt = joblib.load("gbdt_model.model")  
print(load_rf.predict_proba(X_test[0:5]))
print(load_gbdt.predict_proba(X_test[0:5]))
