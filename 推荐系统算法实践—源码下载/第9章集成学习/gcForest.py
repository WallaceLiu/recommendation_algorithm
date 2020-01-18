
# coding: utf-8

# ## 1）环境设定

# In[31]:

import argparse
import numpy as np
import sys
from keras.datasets import mnist
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
sys.path.insert(0, "lib")

from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json

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


# ## 2）数据准备

# In[32]:

# 数据处理，读取libSVM格式数据，并且将数据归一化，样本划分，并且根据batch参数生成batch
def process_data(data_path, feature_size, test_rat=0.3, random_seed=0, train_batch_size=5000, test_batch_size=5000):
    # 读取文件
    # batch生成
    return {"train_batch": train_batch, "test_batch": test_batch, "X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}

# In[40]:

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

X_train = np.array(data['X_train'])
y_train = np.array(data['Y_train']).reshape(-1,) 
X_test = np.array(data['X_test'])
y_test = np.array(data['Y_test']).reshape(-1,) 

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape


# In[41]:

print X_train[0:2]
print y_train[0:2]

# (a, b), (c, d) = mnist.load_data()
# print a.shape
# print b.shape
# print c.shape
# print d.shape
# print a[0:2]
# print b[0:2]


# ## 3）gcForest模型

# In[49]:

# 模型参数
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 2, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 2, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 2, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


# In[54]:

# 模型参数
config = get_toy_config()

# 模型初始化    
gc = GCForest(config)

# 模型训练
X_train_enc = gc.fit_transform(X_train, y_train)
    
# 模型预测
y_pred = gc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))

#  xgboost/RF预测分类.
X_test_enc = gc.transform(X_test)
X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
X_train_origin = X_train.reshape((X_train.shape[0], -1))
X_test_origin = X_test.reshape((X_test.shape[0], -1))
X_train_enc = np.hstack((X_train_origin, X_train_enc))
X_test_enc = np.hstack((X_test_origin, X_test_enc))
print("X_train_enc.shape={}, X_test_enc.shape={}".format(X_train_enc.shape, X_test_enc.shape))
clf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)
clf.fit(X_train_enc, y_train)
y_pred = clf.predict(X_test_enc)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy of Other classifier using gcforest's X_encode = {:.2f} %".format(acc * 100))


# In[55]:

#  计算AUC指标
probs_train= clf.predict_proba(X_train_enc)  
AUC1 = metrics.roc_auc_score(y_train, probs_train[:,1])
print("Train Auc: %s"%(AUC1))
    
probs_test= clf.predict_proba(X_test_enc)  
AUC2 = metrics.roc_auc_score(y_test, probs_test[:,1])
print("Test Auc: %s"%(AUC2))

#     # dump
#     with open("test.pkl", "wb") as f:
#         pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
#     # load
#     with open("test.pkl", "rb") as f:
#         gc = pickle.load(f)
#     y_pred = gc.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print("Test Accuracy of GcForest (save and load) = {:.2f} %".format(acc * 100))


# In[ ]:

# dump
with open("test.pkl", "wb") as f:
    pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
# load
with open("test.pkl", "rb") as f:
    gc = pickle.load(f)
y_pred = gc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy of GcForest (save and load) = {:.2f} %".format(acc * 100))

