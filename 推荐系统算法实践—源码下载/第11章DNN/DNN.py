
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

# 数据处理
def process_data(data_type, my_path, feature_size, batch_size=32, num_epochs=1):
    filenames = get_file_list(my_path)
    next_element = read_my_file_format(data_type, filenames, feature_size, batch_size, num_epochs)
    return next_element

# 创建session，指定GPU或者CPU使用率
def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    # server = tf.train.Server.create_local_server()
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# In[3]:

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


# ## 3）DNN模型

# In[4]:

class DNN(object):
    """ 初始化成员变量 """
    def __init__(self, 
                 feature_size, 
                 loss_fuc, 
                 train_optimizer, 
                 learning_rate, 
                 reg_type, 
                 reg_param, 
                 dnn_layer, 
                 dnn_active_fuc, 
                 is_dropout_dnn, 
                 dropout_dnn, 
                 is_batch_norm):
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
        # dnn_layer
        self.dnn_layer = dnn_layer
        self.dnn_active_fuc = dnn_active_fuc
        # dropout_dnn
        self.is_dropout_dnn = is_dropout_dnn
        self.dropout_dnn = dropout_dnn  
        # is_batch_norm
        self.is_batch_norm = is_batch_norm
        
        # aglobal_step
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    """ dnn全连接层计算 """    
    def _udf_full_connect(self, inputs, input_size, output_size, activation='relu'):
        # 生成或者攻取weights和biases
        weights = tf.get_variable("weights",
                                    [input_size, output_size],
                                     initializer=tf.glorot_normal_initializer(),
                                    trainable=True)
        biases = tf.get_variable("biases",
                                    [output_size],
                                    initializer=tf.glorot_normal_initializer(),
                                    trainable=True)
        # 全连接计算
        layer = tf.matmul(inputs, weights) + biases
        # 激活函数
        if activation == 'relu':
            layer = tf.nn.relu(layer)
        elif activation == 'tanh':
            layer = tf.nn.tanh(layer)
        return layer        
        
    def train(self, batch_data, is_train):
        """ 1 定义输入数据 """
        print("1 定义输入数据")
        with tf.name_scope('input_data'):
            # 标签：[batch_size, 1]
            labels = batch_data['labels']
            # 用户特征向量：[batch_size, feature_size]
            dense_vector = tf.reshape(batch_data['dense_vector'], shape=[-1, feature_size]) # None * feature_size * 1
            print("%s: %s" % ("dense_vector", dense_vector))
            print("%s: %s" % ("labels", labels))
            
        """ 2 定义网络输出 """
        print("2 DNN网络输出" )
        with tf.name_scope("DNN_Comput_Score"):
            # 第一层计算
            with tf.variable_scope("deep_layer1", reuse=tf.AUTO_REUSE):
                input_size = self.feature_size
                output_size = self.dnn_layer[0]
                deep_inputs = dense_vector # None * F
                print("%s: %s" % ("deep_layer1, deep_inputs", deep_inputs))
                # 输入dropout
                if is_train and self.is_dropout_dnn:
                    deep_inputs = tf.nn.dropout(deep_inputs, self.dropout_dnn[0])
                # 全连接计算    
                deep_outputs = self._udf_full_connect(deep_inputs, input_size, output_size, self.dnn_active_fuc[0])
                # batch_norm
                if self.is_batch_norm:
                    deep_outputs = tf.layers.batch_normalization(deep_outputs, axis=-1, training=is_train) 
                # 输出dropout
                if is_train and self.is_dropout_dnn:
                    deep_outputs = tf.nn.dropout(deep_outputs, dropout_dnn[1])
                print("%s: %s" % ("deep_layer1, deep_outputs", deep_outputs))     
            # 中间层计算
            for i in range(len(self.dnn_layer) - 1):
                with tf.variable_scope("deep_layer%d"%(i+2), reuse=tf.AUTO_REUSE):
                    # 全连接计算
                    deep_outputs = self._udf_full_connect(deep_outputs, self.dnn_layer[i], self.dnn_layer[i+1], self.dnn_active_fuc[i+1])  
                    # batch_norm
                    if self.is_batch_norm:
                        deep_outputs = tf.layers.batch_normalization(deep_outputs, axis=-1, training=is_train)
                    # 输出dropout  
                    if is_train and self.is_dropout_dnn:
                        deep_outputs = tf.nn.dropout(deep_outputs, self.dropout_dnn[i+2])
                    print("%s, %s: %s" % ("deep_layer%d"%(i+2), "deep_outputs", deep_outputs))     
            # 输出层计算   
            with tf.variable_scope("deep_layer%d"%(len(dnn_layer)+1), reuse=tf.AUTO_REUSE):
                deep_outputs = self._udf_full_connect(deep_outputs, self.dnn_layer[-1], 1, self.dnn_active_fuc[-1])        
            print("%s, %s: %s" % ("deep_layer%d"%(len(dnn_layer)+1), "deep_outputs", deep_outputs))
            # 正则化，默认L2
            dnn_regularization = 0.0
            for j in range(len(self.dnn_layer)+1):        
                with tf.variable_scope("deep_layer%d"%(j+1), reuse=True):
                    weights = tf.get_variable("weights")
                    if reg_type == 'l1_reg':
                        dnn_regularization = dnn_regularization + tf.reduce_sum(tf.abs(weights))
                    elif reg_type == 'l2_reg':
                        dnn_regularization = dnn_regularization + tf.nn.l2_loss(weights)
                    else:  
                        dnn_regularization = dnn_regularization + tf.nn.l2_loss(weights)
        # Deep输出            
        Y_Out=deep_outputs
        print("%s: %s" % ("Y_Out", Y_Out))
        # ---------- score ----------  
        score=tf.nn.sigmoid(Y_Out,name='score')
        score=tf.reshape(score, shape=[-1, 1])
        print("%s: %s" % ("score", score))
        
        """ 3 定义损失函数和AUC指标 """
        print("3 定义损失函数和AUC指标" ) 
        with tf.name_scope("loss"):
            # loss：Squared_error，Cross_entropy ,FTLR
            regularization = self.reg_param * dnn_regularization            
            if loss_fuc == 'Squared_error':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
            elif loss_fuc == 'Cross_entropy':
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(Y_Out, [-1]), labels=tf.reshape(labels, [-1]))) + regularization
            elif loss_fuc == 'FTLR':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
            # AUC                  
            auc = tf.metrics.auc(labels, score)
            print("%s: %s" % ("labels", labels))
            
        """ 4 设定optimizer """
        print("4 设定optimizer" ) 
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
        print("5 设定summary" ) 
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accumulate_auc", auc[0])
            for j in range(len(self.dnn_layer)+1):        
                with tf.variable_scope("deep_layer%d"%(j+1), reuse=True):
                    weights = tf.get_variable("weights")
                    tf.summary.histogram("w%d"%(j+1), weights)
            # 好几个summary，所以这里要merge_all
            summary_op = tf.summary.merge_all()
            
        """6 返回结果 """
        return Y_Out, score, regularization, loss, auc, train_step, labels, score, summary_op


# ## 4）模型训练测试

# In[5]:

# 数据参数
print("0 数据准备和参数设置" ) 
filenames = '/data/csv-all'
data_type='csv'
feature_size = 530
batch_size = 60000
num_epochs = 200
next_element = process_data(data_type, filenames, feature_size, batch_size, num_epochs)
print("%s: %s" % ("next_element", next_element))
    
# 模型参数   
loss_fuc = 'Squared_error'
train_optimizer = 'Adam'
learning_rate = 0.01
reg_type = 'l2_reg'
reg_param = 0.0

dnn_layer=[100,50]
dnn_active_fuc=['relu','relu','output']
dropout_fm=[1,1]
is_dropout_dnn=True
dropout_dnn=[0.7,0.7,0.7]
is_batch_norm=True

log_path='/data/log/DNN_Squared_error_L2_0_20180816_01'

# 开始训练
bea_model = DNN(feature_size, 
                 loss_fuc, 
                 train_optimizer, 
                 learning_rate, 
                 reg_type, 
                 reg_param, 
                 dnn_layer, 
                 dnn_active_fuc, 
                 is_dropout_dnn, 
                 dropout_dnn, 
                 is_batch_norm)
Y_Out, score, regularization, loss, auc, train_step, labels, score, summary_op = bea_model.train(next_element, is_train=True)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
gpu_fraction = 0.5
my_device='/gpu:0'
with tf.device(my_device):
    sess = get_session(gpu_fraction)
    sess.run(init_op)
    batch_cnt = 0
    #选定可视化存储目录
    writer = tf.summary.FileWriter(log_path, sess.graph)
    print("6 迭代过程" ) 
    try:
         while True:
            batch_cnt = batch_cnt + 1           
            a, b, c, summary = sess.run([loss, auc, train_step, summary_op])
            if batch_cnt % 100 == 0 or batch_cnt <= 10:
                y, p = sess.run([labels, score])
                if y.sum() > 0.0:
                    batch_auc=metrics.roc_auc_score(y, p)
                else:
                    batch_auc=0.0
                print("batch: {} loss: {:.4f} accumulate_auc: {:.4f} batch_auc: {:.4f}".format(batch_cnt, a, b[0], batch_auc))
                writer.add_summary(summary, batch_cnt)
    except tf.errors.OutOfRangeError:
        print("Train end of dataset")   
