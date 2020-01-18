
# coding: utf-8

# ## 1）环境准备

In [1]:
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import math
import re

from sklearn import preprocessing
from os import path, listdir
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.contrib import layers

from sklearn import metrics

import time
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print tf.__version__
print tf.__path__

In [6]:
"""
解析CSV格式，对输入的每一行样本进行格式解析，返回labels和dense_vector格式数据
例如输入CSV格式字符串： 0.0,0.6666666666666666,0.5,0.0,0.0,0.0,0.0,0.7272727272727273,0.42857142857142855
函数参数：
line：需要解析的字符串
feature_size：特征长度
函数返回：
返回字典，格式：{'labels': labels, 'dense_vector': dense_vector}
labels：样本的labels
dense_vector：样本的特征向量
"""

In [8]:
# 测试数据
filenames = '/data/all-csv'
feature_size = 530
batch_size = 3
num_epochs = 1
data_type = 'csv'
next_element = process_data(data_type, filenames, feature_size, batch_size, num_epochs)
print next_element['dense_vector']
print next_element['labels']

gpu_fraction = 0.2
my_device='/gpu:0'
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
with tf.device(my_device):
    sess = get_session(gpu_fraction)
    sess.run(init_op)
    dense_vector, labels = sess.run([next_element['dense_vector'],next_element['labels']])
    print dense_vector
    print labels

In [7]:
#基于逻辑回归的网络结构在TensorFlow中实现逻辑回归模型。其中“LR模型”代码省略，具体内容可以参考6.2.3节中的相关代码。
class LR(object):
    """ 初始化成员变量 """
    def __init__(self, feature_size, loss_fuc, train_optimizer, learning_rate, reg_type, reg_param):
        # 特征向量长度
        self.feature_size = feature_size
        # 损失函数
        self.loss_fuc = loss_fuc
        # 优化方法
        self.train_optimizer = train_optimizer
        # 学习率
        self.learning_rate = learning_rate
        # 正则类型
        self.reg_type = reg_type
        # 正则因子
        self.reg_param = reg_param
        # aglobal_step
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def train(self, batch_data):
        """ 1 定义输入数据 """
        with tf.name_scope('input_data'):


In [9]:
# 数据准备  
filenames = '/data/csv-all'
data_type='csv'
feature_size = 530
batch_size = 60000
num_epochs = 200
next_element = process_data(data_type, filenames, feature_size, batch_size, num_epochs)
    
# 模型参数   
loss_fuc = 'Squared_error'
train_optimizer = 'Adam'
learning_rate = 0.01
reg_type = 'l2_reg'
reg_param = 0.0
log_path='/data/log/Squared_error_lr_L2_0_20180816_01'

# 开始训练
bea_model = LR(feature_size, loss_fuc, train_optimizer, learning_rate, reg_type, reg_param)
Y_Out, score, regularization, loss, auc, train_step, w_zero_ratio, w_avg, labels, score, summary_op = bea_model.train(next_element)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
gpu_fraction = 0.4
my_device='/gpu:0'
with tf.device(my_device):
    sess = get_session(gpu_fraction)
    sess.run(init_op)
    batch_cnt = 0
    #选定可视化存储目录
    writer = tf.summary.FileWriter(log_path, sess.graph)
    try:
         while True:
            batch_cnt = batch_cnt + 1           
            a, b, c, d, e, summary = sess.run([loss, auc, w_zero_ratio, w_avg, train_step, summary_op])
            if batch_cnt % 50 == 0 or batch_cnt <= 10:
                y, p = sess.run([labels, score])
                if y.sum() > 0.0:
                    batch_auc=metrics.roc_auc_score(y, p)
                else:
                    batch_auc=0.0
                print("batch: {} loss: {:.4f} accumulate_auc: {:.4f} batch_auc: {:.4f} w_zero_ratio: {:.4f} w_avg: {:.4f}".format(batch_cnt, a, b[0], batch_auc, c, d))
                writer.add_summary(summary, batch_cnt)
    except tf.errors.OutOfRangeError:
        print("3、Train end of dataset")   

