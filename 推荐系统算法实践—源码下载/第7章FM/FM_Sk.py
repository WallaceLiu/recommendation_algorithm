
# coding: utf-8

# ## 0）环境设定

# In[1]:

from sklearn import metrics
from os import path, listdir
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib  
from sklearn import preprocessing
from sklearn import metrics
from fastFM import als
import numpy as np
import pandas as pd
import random


# ## 1）数据准备

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


# In[6]:

print Y_train


# ## 3）FM模型

# In[4]:

# 3.1 建立FM模型，并且设定参数  
fm_model = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=10, random_state=0, l2_reg_w=0.0, l2_reg_V=0.0, l2_reg=0)

# 3.2 训练FM模型
fm_model.fit(X_train,Y_train)

# 3.3 采用测试集验证模型离线指标  
# 训练集AUC
probs_train= fm_model.predict_proba(X_train)  
AUC1 = metrics.roc_auc_score(Y_train, probs_train[:,1])
print("Train Auc: %s"%(AUC1))

# 测试集AUC
probs_test= fm_model.predict_proba(X_test)  
predict_test = fm_model.predict(X_test)
AUC2 = metrics.roc_auc_score(Y_test, probs_test[:,1])
print("Test Auc: %s"%(AUC2))

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
joblib.dump(fm_model,"FM.model")  
#模型加载
print("模型加载")
load_lr = joblib.load("FM.model")    
print(load_lr.predict_proba(X_test[0:5]))


# In[ ]:

# 3.4 打印模型参数
print("参数:",lr_model.coef_)  
print("截距:",lr_model.intercept_)  
print("稀疏化特征比率:%.2f%%" %(np.mean(lr_model.coef_.ravel()==0)*100))  
print("=========sigmoid函数转化的值，即：概率p=========")  
print(lr_model.predict_proba(X_test[0:5]))     #sigmoid函数转化的值，即：概率p  

# 3.5 模型保存
joblib.dump(fm_model,"FM.model")  
#模型加载
load_lr = joblib.load("FM.model")    
print(load_lr.predict_proba(X_test[0:5]))

