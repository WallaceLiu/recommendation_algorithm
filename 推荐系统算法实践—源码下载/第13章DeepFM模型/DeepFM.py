
# coding: utf-8

# ## 0）环境准备

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


# ## 1）数据准备Dataset格式

# In[2]:

# 每一行解析，解析标签csv格式
# 0.0,0.6666666666666666,0.5,0.0,0.0,0.0,0.0,0.7272727272727273,0.42857142857142855

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


# ## 2）DeepFM模型

# In[4]:

class DeepFM(object):
    """ 初始化成员变量 """
    def __init__(self, 
                 feature_size,
                 model_type='deep_fm',                 
                 fm_v_size=32, 
                 reg_type='l2_reg',
                 reg_w=0.0,
                 reg_v=0.0,
                 reg_dnn=0.0,
                 dnn_layer=[64,64],
                 dnn_active_fuc=['relu','relu','output'],
                 is_batch_norm=False,
                 is_dropout_fm=False,
                 dropout_fm=[1,1],
                 is_dropout_dnn=False,
                 dropout_dnn=[0.5,0.5,0.5],
                 out_lay_type='line',
                 loss_fuc='Squared_error', 
                 train_optimizer='Adam', 
                 learning_rate=0.01):
        # 特征向量长度
        self.feature_size = feature_size
        # 支模型类型持：lr, fm, dnn, deep_fm
        self.model_type = model_type
        # fm层大小
        self.fm_v_size = fm_v_size
        # 正则类型
        self.reg_type = reg_type
        # 正则因子，lr层，fm层，dnn层
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.reg_dnn = reg_dnn
        # dnn参数
        self.dnn_layer = dnn_layer
        self.dnn_active_fuc = dnn_active_fuc
        self.is_batch_norm = is_batch_norm
        self.is_dropout_fm = is_dropout_fm
        self.dropout_fm = dropout_fm
        self.is_dropout_dnn = is_dropout_dnn
        self.dropout_dnn = dropout_dnn
        # 输出层
        self.out_lay_type = out_lay_type        
        # 损失函数
        self.loss_fuc = loss_fuc
        # 优化方法
        self.train_optimizer = train_optimizer
        # 学习率
        self.learning_rate = learning_rate
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
        
    def train(self, batch_data, is_train=True):
        """ 1 定义输入数据 """
        print("1 定义输入数据" )
        with tf.name_scope('input_data'):
            # 标签：[batch_size, 1]
            labels = batch_data['labels']
            # 用户特征向量：[batch_size, feature_size]
            dense_vector = tf.reshape(batch_data['dense_vector'], shape=[-1, self.feature_size]) # None * feature_size
            print("%s: %s" % ("dense_vector", dense_vector))
            print("%s: %s" % ("labels", labels))
            
        """ 2 FM层网络输出 """
        print("2 FM层网络输出" )
        with tf.name_scope("FM"):
            # FM参数，生成或者获取W V
            with tf.variable_scope("fm_layer", reuse=tf.AUTO_REUSE):
                self.FM_W = tf.get_variable(name='fm_w', shape=[self.feature_size, 1], initializer=tf.glorot_normal_initializer())
                self.FM_V = tf.get_variable(name='fm_v', shape=[self.feature_size, self.fm_v_size], initializer=tf.glorot_normal_initializer())    
            print("%s: %s" % ("FM_W", self.FM_W))
            print("%s: %s" % ("FM_V", self.FM_V))
        
             # 输入样本准备
            Input_x = tf.reshape(dense_vector, shape=[-1, self.feature_size, 1]) # None * feature_size 
            print("%s: %s" % ("Input_x", Input_x))

            # ---------- W * X ----------   
            Y_first = tf.reduce_sum(tf.multiply(self.FM_W, Input_x), 2)  # None * F
            ## 增加dropout，防止过拟合
            if is_train and self.is_dropout_fm:
                Y_first = tf.nn.dropout(Y_first, self.dropout_fm[0]) # None * F        
            print("%s: %s" % ("Y_first", Y_first))

            # ---------- Vij * Wij ---------------
            # sum_square part
            embeddings = tf.multiply(self.FM_V, Input_x) # None * V * X 
            print("%s: %s" % ("embeddings", embeddings))
        
            summed_features_emb = tf.reduce_sum(embeddings, 1)  # sum(v*x)
            summed_features_emb_square = tf.square(summed_features_emb)  # (sum(v*x))^2

            # square_sum part
            squared_features_emb = tf.square(embeddings) # (v*x)^2
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)   # sum((v*x)^2)

            # second order
            Y_second = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # 0.5*((sum(v*x))^2 - sum((v*x)^2))
            if is_train and self.is_dropout_fm:
                Y_second = tf.nn.dropout(Y_second, self.dropout_fm[1])  # None * K        
            print("%s: %s" % ("Y_second", Y_second))
        
            # 正则化，默认L2
            if reg_type == 'l1_reg':
                lr_regularization = tf.reduce_sum(tf.abs(self.FM_W))
                fm_regularization = tf.reduce_sum(tf.abs(self.FM_V))
            elif reg_type == 'l2_reg':
                lr_regularization = tf.nn.l2_loss(self.FM_W)
                fm_regularization = tf.nn.l2_loss(self.FM_V)
            else:  
                lr_regularization = tf.nn.l2_loss(self.FM_W)
                fm_regularization = tf.nn.l2_loss(self.FM_V)
            
        """ 3 Deep层网络输出 """
        print("3 Deep层网络输出" )
        with tf.name_scope("Deep"):
            # 第一层计算
            print("lay%s, input_size: %s, output_size: %s, active_fuc: %s" % (1, self.feature_size*self.fm_v_size, self.dnn_layer[0], self.dnn_active_fuc[0]))
            with tf.variable_scope("deep_layer1", reuse=tf.AUTO_REUSE):
                input_size = self.feature_size*self.fm_v_size
                output_size = self.dnn_layer[0]
                deep_inputs = tf.reshape(embeddings, shape=[-1, input_size]) # None * (F*K)
                print("%s: %s" % ("lay1, deep_inputs", deep_inputs))
                # 输入dropout
                if is_train and self.is_dropout_dnn:
                    deep_inputs = tf.nn.dropout(deep_inputs, self.dropout_dnn[0])
                # 全连接计算    
                deep_outputs = self._udf_full_connect(deep_inputs, input_size, output_size, self.dnn_active_fuc[0])
                print("%s: %s" % ("lay1, deep_outputs", deep_outputs))
                # batch_norm
                if self.is_batch_norm:
                    deep_outputs = tf.layers.batch_normalization(deep_outputs, axis=-1, training=is_train) 
                # 输出dropout
                if is_train and self.is_dropout_dnn:
                    deep_outputs = tf.nn.dropout(deep_outputs, dropout_dnn[1])
            # 中间层计算
            for i in range(len(self.dnn_layer) - 1):
                with tf.variable_scope("deep_layer%d"%(i+2), reuse=tf.AUTO_REUSE):
                    print("lay%s, input_size: %s, output_size: %s, active_fuc: %s" % (i+2, self.dnn_layer[i], self.dnn_layer[i+1], self.dnn_active_fuc[i+1]))
                    # 全连接计算
                    deep_outputs = self._udf_full_connect(deep_outputs, self.dnn_layer[i], self.dnn_layer[i+1], self.dnn_active_fuc[i+1])
                    print("lay%s, deep_outputs: %s" % (i+2, deep_outputs))
                    # batch_norm
                    if self.is_batch_norm:
                        deep_outputs = tf.layers.batch_normalization(deep_outputs, axis=-1, training=is_train)
                    # 输出dropout  
                    if is_train and self.is_dropout_dnn:
                        deep_outputs = tf.nn.dropout(deep_outputs, self.dropout_dnn[i+2])
            # 输出层计算
            print("lay_last, input_size: %s, output_size: %s, active_fuc: %s" % (self.dnn_layer[-1], 1, self.dnn_active_fuc[-1]))
            with tf.variable_scope("deep_layer%d"%(len(dnn_layer)+1), reuse=tf.AUTO_REUSE):
                deep_outputs = self._udf_full_connect(deep_outputs, self.dnn_layer[-1], 1, self.dnn_active_fuc[-1])
                print("lay_last, deep_outputs: %s" % (deep_outputs))
        
            # 正则化，默认L2
            dnn_regularization = 0.0
            for j in range(len(self.dnn_layer)+1):        
                with tf.variable_scope("deep_layer%d"%(j+1), reuse=True):
                    weights = tf.get_variable("weights")
                    if self.reg_type == 'l1_reg':
                        dnn_regularization = dnn_regularization + tf.reduce_sum(tf.abs(weights))
                    elif self.reg_type == 'l2_reg':
                        dnn_regularization = dnn_regularization + tf.nn.l2_loss(weights)
                    else:  
                        dnn_regularization = dnn_regularization + tf.nn.l2_loss(weights)
        # Deep输出            
        Y_deep=deep_outputs
        print("%s: %s" % ("Y_deep", Y_deep))
        
        """ 4 DeepFM层网络输出 """
        print("4 DeepFM层网络输出" )
        # ---------- DeepFM ----------
        with tf.name_scope("Deep_FM"):
            # 最后一层的输入层准备
            concat_input = tf.concat([Y_first, Y_second, Y_deep], axis=1)
            if self.model_type=="deep_fm":
                concat_input = tf.concat([Y_first, Y_second, Y_deep], axis=1)
                print("%s: %s" % ("concat_input", concat_input))
                input_size = self.feature_size + self.fm_v_size + self.dnn_layer[-1]
                regularization = self.reg_w*lr_regularization + self.reg_v*fm_regularization + self.reg_dnn*dnn_regularization
            elif self.model_type=="fm":
                concat_input = tf.concat([Y_first, Y_second], axis=1)
                print("%s: %s" % ("concat_input", concat_input))
                input_size = self.feature_size + self.fm_v_size 
                regularization = self.reg_w*lr_regularization + self.reg_v*fm_regularization
            elif self.model_type=="dnn":
                concat_input = Y_deep
                print("%s: %s" % ("concat_input", concat_input))
                input_size =  self.dnn_layer[-1]
                regularization = self.reg_dnn*dnn_regularization
            elif self.model_type=="lr":
                concat_input = tf.concat([Y_first], axis=1)
                print("%s: %s" % ("concat_input", concat_input))
                input_size = self.feature_size
                regularization = self.reg_w*lr_regularization
            else:
                concat_input = tf.concat([Y_first, Y_second, Y_deep], axis=1)
                print("%s: %s" % ("concat_input", concat_input))
                input_size = self.feature_size + self.fm_v_size + self.dnn_layer[-1]
                regularization = self.reg_w*lr_regularization + self.reg_v*fm_regularization + self.reg_dnn*dnn_regularization
                
            # 最后一层的输出，采用w*concat_input + b 全连接 ,也可以直接对concat_input进行sum求和  
            with tf.variable_scope("deepfm_out", reuse=tf.AUTO_REUSE):
                self.DF_W = tf.get_variable(name='df_w', shape=[input_size, 1], initializer=tf.glorot_normal_initializer())
                self.DF_B = tf.get_variable(name='df_bias', shape=[1], initializer=tf.constant_initializer(0.0))
            print("%s: %s" % ("DF_W", self.DF_W))
            print("%s: %s" % ("DF_B", self.DF_B))
            print("%s: %s" % ("out_lay_type", self.out_lay_type))
            if self.out_lay_type=="line":
                Y_sum = tf.reduce_sum(concat_input, 1)  # None * 1
                print("%s: %s" % ("Y_sum", Y_sum))
                Y_bias = self.DF_B * tf.ones_like(Y_sum, dtype=tf.float32)
                print("%s: %s" % ("Y_bias", Y_bias))
                Y_Out = tf.add(Y_sum, Y_bias, name='Y_Out')
            elif self.out_lay_type=="matmul":
                Y_Out = tf.add(tf.matmul(concat_input, self.DF_W), self.DF_B, name='Y_Out')
            else:
                Y_sum = tf.reduce_sum(concat_input, 1)  # None * 1
                print("%s: %s" % ("Y_sum", Y_sum))
                Y_bias = self.DF_B * tf.ones_like(Y_sum, dtype=tf.float32)
                print("%s: %s" % ("Y_bias", Y_bias))
                Y_Out = tf.add(Y_sum, Y_bias, name='Y_Out')
            print("%s: %s" % ("Y_Out", Y_Out))
        score=tf.nn.sigmoid(Y_Out,name='score')
        score=tf.reshape(score, shape=[-1, 1])
        print("%s: %s" % ("score", score))
        
        """ 5 定义损失函数和AUC指标 """
        print("5 定义损失函数和AUC指标" ) 
        with tf.name_scope("loss"):
            # loss：Squared_error，Cross_entropy ,FTLR
            if self.loss_fuc == 'Squared_error':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
            elif self.loss_fuc == 'Cross_entropy':
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(Y_Out, [-1]), labels=tf.reshape(labels, [-1]))) + regularization
            elif self.loss_fuc == 'FTLR':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
            # AUC                  
            auc = tf.metrics.auc(labels, score)
            print("%s: %s" % ("labels", labels))
            
        """ 6 设定optimizer """
        print("6 设定optimizer" ) 
        with tf.name_scope("optimizer"):
            with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
                #------bulid optimizer------
                if self.train_optimizer == 'Adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
                elif self.train_optimizer == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
                elif self.train_optimizer == 'Momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
                elif self.train_optimizer == 'ftrl':
                    optimizer = tf.train.FtrlOptimizer(learning_rate)
                train_step = optimizer.minimize(loss, global_step=self.global_step)               

        """7 设定summary，以便在Tensorboard里进行可视化 """
        print("7 设定summary" ) 
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accumulate_auc", auc[0])
            tf.summary.histogram("FM_W", self.FM_W)
            tf.summary.histogram("FM_V", self.FM_V)
            for j in range(len(self.dnn_layer)+1):        
                with tf.variable_scope("deep_layer%d"%(j+1), reuse=True):
                    weights = tf.get_variable("weights")
                    tf.summary.histogram("dnn_w_%d"%(j+1), weights)
            # 好几个summary，所以这里要merge_all
            summary_op = tf.summary.merge_all()
            
        """8 返回结果 """
        return Y_Out, score, regularization, loss, auc, train_step, labels, score, summary_op


# ## 3）模型训练测试

# In[5]:

# 数据参数
print("0 数据准备和参数设置" ) 
filenames = '/data/csv-all'
data_type='csv'
feature_size = 530
fm_v_size = 20
batch_size = 10000
num_epochs = 100
next_element = process_data(data_type, filenames, feature_size, batch_size, num_epochs)
print("%s: %s" % ("next_element", next_element))
    
# 模型参数   
loss_fuc = 'Squared_error'
train_optimizer = 'Adam'
learning_rate = 0.01
reg_type = 'l2_reg'
reg_w = 0.0
reg_v = 0.0
reg_dnn = 0.0

model_type = 'deep_fm'
dnn_layer=[64,32]
dnn_active_fuc=['relu','relu','output']
is_batch_norm=True
is_dropout_fm=False
dropout_fm=[1,1]
is_dropout_dnn=True
dropout_dnn=[0.7,0.7,0.7]
out_lay_type='line'

log_path='/data/log/DeepFM_Squared_error_L2_0_20180816_01'

# 开始训练
bea_model = DeepFM(feature_size,
                 model_type,                 
                 fm_v_size, 
                 reg_type,
                 reg_w,
                 reg_v,
                 reg_dnn,
                 dnn_layer,
                 dnn_active_fuc,
                 is_batch_norm,
                 is_dropout_fm,
                 dropout_fm,
                 is_dropout_dnn,
                 dropout_dnn,
                 out_lay_type,
                 loss_fuc, 
                 train_optimizer, 
                 learning_rate)
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
    print("8 迭代过程" ) 
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
