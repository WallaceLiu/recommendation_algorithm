
# coding: utf-8

# ## 1）环境设定

# In[25]:

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
import platform
print("Python Version: %s"%(platform.python_version()))


# ## 2）数据准备

# In[26]:

"""
处理libSVM数据方法，生成样本，支持Batch格式返回，也支持X/Y返回
"""
def process_data(data_path, feature_size, test_rat=0.3, random_seed=0, train_batch_size=5000, test_batch_size=5000):
    # 读取文件
    # batch生成
    return {"train_batch": train_batch, "test_batch": test_batch, "X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}

# In[27]:

# 数据测试
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

# 查看样本数据大小
print("X_train.shape: ")
print(X_train.shape)
print("Y_train.shape: ")
print(Y_train.shape)
print("X_test.shape: ")
print(X_test.shape)
print("Y_test.shape: ")
print(Y_test.shape)


# ## 3）LR模型

# In[30]:

# 3.1 建立逻辑回归模型，并且设定参数  
lr_model= LogisticRegression(penalty='l2', C=1000, solver='lbfgs', max_iter=500)

# 3.2 训练逻辑回归模型
lr_model.fit(X_train,Y_train.values.ravel())


# In[31]:

# 3.3 采用测试集验证模型离线指标  
# 训练集AUC
probs_train= lr_model.predict_proba(X_train)  
AUC1 = metrics.roc_auc_score(Y_train, probs_train[:,1])
print("Train Auc: %s"%(AUC1))

# 测试集AUC
probs_test= lr_model.predict_proba(X_test)  
predict_test = lr_model.predict(X_test)
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


# In[42]:

# 3.4 打印模型参数
w=lr_model.coef_
print("参数大小:")
print(w.shape)
print("参数前10个:")
print(lr_model.coef_[:,0:10]) 
print("截距:")  
print(lr_model.intercept_) 
print("稀疏化特征比率:%.2f%%" %(np.mean(lr_model.coef_.ravel()==0)*100))  
print("sigmoid函数转化的值，即：概率p")  
print(lr_model.predict_proba(X_test[0:5]))


# In[43]:

# 3.5 模型保存
joblib.dump(lr_model,"logistic_lr.model")  
#模型加载
load_lr = joblib.load("logistic_lr.model")    
print(load_lr.predict_proba(X_test[0:5]))


# In[ ]:

# 3.1 建立逻辑回归模型，并且设定参数  
lr_model= LogisticRegression(penalty='l2', C=1000, solver='lbfgs', max_iter=500)

# 3.2 训练逻辑回归模型
lr_model.fit(X_train,Y_train)

# 3.3 采用测试集验证模型离线指标  
# 训练集AUC
probs_train= lr_model.predict_proba(X_train)  
AUC1 = metrics.roc_auc_score(Y_train, probs_train[:,1])
print("Train Auc: %s"%(AUC1))

# 测试集AUC
probs_test= lr_model.predict_proba(X_test)  
predict_test = lr_model.predict(X_test)
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

# 3.4 打印模型参数
print("参数:",lr_model.coef_)  
print("截距:",lr_model.intercept_)  
print("稀疏化特征比率:%.2f%%" %(np.mean(lr_model.coef_.ravel()==0)*100))  
print("=========sigmoid函数转化的值，即：概率p=========")  
print(lr_model.predict_proba(X_test[0:5]))     #sigmoid函数转化的值，即：概率p  

# 3.5 模型保存
joblib.dump(lr_model,"logistic_lr.model")  
#模型加载
load_lr = joblib.load("logistic_lr.model")    
print(load_lr.predict_proba(X_test[0:5]))

# In[30]:

# 3.1 建立逻辑回归模型，并且设定参数  
lr_model= LogisticRegression(penalty='l2', C=1000, solver='lbfgs', max_iter=500)

# 3.2 训练逻辑回归模型
lr_model.fit(X_train,Y_train)


# In[46]:

# 3.3 采用测试集验证模型离线指标  
# 训练集AUC
probs_train= lr_model.predict_proba(X_train)  
AUC1 = metrics.roc_auc_score(Y_train, probs_train[:,1])
print("Train Auc: %s"%(AUC1))

# 测试集AUC
probs_test= lr_model.predict_proba(X_test)  
predict_test = lr_model.predict(X_test)
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


# In[49]:

# 3.4 打印模型参数
print("参数:",lr_model.coef_)  
print("截距:",lr_model.intercept_)  
print("稀疏化特征比率:%.2f%%" %(np.mean(lr_model.coef_.ravel()==0)*100))  
print("=========sigmoid函数转化的值，即：概率p=========")  
print(lr_model.predict_proba(X_test[0:5]))     #sigmoid函数转化的值，即：概率p  


# In[53]:

# 3.5 模型保存
joblib.dump(lr_model,"logistic_lr.model")  
#模型加载
load_lr = joblib.load("logistic_lr.model")    
print(load_lr.predict_proba(X_test[0:5]))
