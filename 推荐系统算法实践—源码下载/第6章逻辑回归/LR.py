
# coding: utf-8

# ## 1）环境准备

# In[1]:

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


# ## 2）数据准备Dataset格式

# In[6]:

"""
解析CSV格式，对输入的每一行样本，进行格式解析，返回labels和dense_vector格式数据
例如输入csv格式字符串： 0.0,0.6666666666666666,0.5,0.0,0.0,0.0,0.0,0.7272727272727273,0.42857142857142855
"""
# 数据处理
def process_data(data_type, my_path, feature_size, batch_size=32, num_epochs=1):
    filenames = get_file_list(my_path)
    next_element = read_my_file_format(data_type, filenames, feature_size, batch_size, num_epochs)
    return next_element

# 创建session，指定GPU或者CPU使用率
def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# In[8]:

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


# ## 3）LR模型

# In[7]:

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
            # 标签：[batch_size, 1]
            labels = batch_data['labels']
            # 用户特征向量：[batch_size, feature_size]
            dense_vector = tf.reshape(batch_data['dense_vector'], shape=[-1, feature_size, 1]) # None * feature_size * 1
            print("%s: %s" % ("dense_vector", dense_vector))
            print("%s: %s" % ("labels", labels))
            
        """ 2 定义网络输出 """
        with tf.name_scope("LR_Comput_Score"):
            # LR参数，生成或者获取w b
            with tf.variable_scope("lr_layer", reuse=tf.AUTO_REUSE):
                self.w = tf.get_variable(name='w', shape=[self.feature_size, 1], initializer=tf.glorot_normal_initializer())
                self.b = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
            print("%s: %s" % ("w", self.w))
            print("%s: %s" % ("b", self.b))
            
            # ---------- w * x  + b----------   
            Y_first = tf.reduce_sum(tf.multiply(self.w, dense_vector), 2)  # None * F
            print("%s: %s" % ("Y_first", Y_first))
            # ---------- sum(w * x)  + b----------   
            Y_Out = tf.reduce_sum(Y_first, 1)
            Y_bias = self.b * tf.ones_like(Y_Out, dtype=tf.float32) # None * 1
            print("%s: %s" % ("Y_bias", Y_bias))
            Y_Out = tf.add(Y_Out, Y_bias, name='Y_Out') 
            print("%s: %s" % ("Y_Out", Y_Out))
            # ---------- score ----------  
            score=tf.nn.sigmoid(Y_Out,name='score')
            score=tf.reshape(score, shape=[-1, 1])
            print("%s: %s" % ("score", score))
        
        """ 3 定义损失函数和AUC指标 """
        with tf.name_scope("loss"):
            # loss：Squared_error，Cross_entropy ,FTLR
            if reg_type == 'l1_reg':
                regularization = self.reg_param * tf.reduce_sum(tf.abs(self.w))
#                 tf.contrib.layers.l1_regularizer(self.reg_param)(self.w) 
            elif reg_type == 'l2_reg':
                regularization = self.reg_param * tf.nn.l2_loss(self.w) 
            else:  
                regularization = self.reg_param * tf.nn.l2_loss(self.w)                 
            
            if loss_fuc == 'Squared_error':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
            elif loss_fuc == 'Cross_entropy':
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(Y_Out, [-1]), labels=tf.reshape(labels, [-1]))) + regularization
            elif loss_fuc == 'FTLR':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
            # AUC                  
            auc = tf.metrics.auc(labels, score)
            print("%s: %s" % ("labels", labels))
            # w为0的比例,w的平均值
            w_zero_ratio = tf.reduce_mean(tf.to_float(tf.abs(self.w) <= 1.0e-5))
            w_avg = tf.reduce_mean(self.w)
            
        """ 4 设定optimizer """
        with tf.name_scope("optimizer"):
            with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
                #------bulid optimizer------
                if train_optimizer == 'Adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
                elif train_optimizer == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
                elif train_optimizer == 'Momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
                elif train_optimizer == 'ftrl':
                    optimizer = tf.train.FtrlOptimizer(learning_rate)
                train_step = optimizer.minimize(loss, global_step=self.global_step)               

        """5 设定summary，以便在Tensorboard里进行可视化 """
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accumulate_auc", auc[0])
            tf.summary.scalar("w_avg", w_avg)
            tf.summary.scalar("w_zero_ratio", w_zero_ratio)
            tf.summary.histogram("w", self.w)
            # 好几个summary，所以这里要merge_all
            summary_op = tf.summary.merge_all()
            
        """6 返回结果 """
        return Y_Out, score, regularization, loss, auc, train_step, w_zero_ratio, w_avg, labels, score, summary_op


# ## 4）模型训练测试

# In[9]:

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
my_device='/gpu:1'
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
