
# coding: utf-8

# In[ ]:


import xgboost as xgb
import numpy as np
import random
import math
import os
import sys
from sklearn import metrics

# 1 数据准备
dtrain = xgb.DMatrix(train_data_path, feature_names = features)
dtest = xgb.DMatrix(test_data_path, feature_names = features)
dvalid = xgb.DMatrix(valid_data_path, feature_names = features)

# 2 参数准备
param = {'booster': booster, 
         'eval_metric':eval_metric, 
         'max_depth':max_depth, 
         'gamma': gamma, 
         'min_child_weight':min_child_weight, 
         'eta':eta, 
         'objective':objective, 
         'subsample': subsample, 
         'colsample_bytree': colsample_bytree}

# 3 模型训练
bst = xgb.train(param, dtrain, round, evals=[(dtrain,'train'), (dtest,'test')])

# 4 模型测试
preds = bst.predict(dtest)
auc = metrics.roc_auc_score(labels, preds)
precision = metrics.average_precision_score(labels, preds)
mae = metrics.mean_absolute_error(labels, preds)
rmse = math.sqrt(metrics.mean_squared_error(labels, preds))

# 5 模型保存
bst.save_model(local_path_bin)
bst.dump_model(local_path) 

