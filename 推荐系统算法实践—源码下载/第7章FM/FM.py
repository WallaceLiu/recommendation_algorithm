
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


# ## 2）FM模型

# In[3]:

class FM(object):
    """ 初始化成员变量 """
    def __init__(self, feature_size, fm_v_size, loss_fuc, train_optimizer, learning_rate, reg_type, reg_param):
        # 特征向量长度
        self.feature_size = feature_size
        # fm_v_size向量长度
        self.fm_v_size = fm_v_size
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
        with tf.name_scope("FM_Comput_Score"):
            # FM参数，生成或者获取W V
            with tf.variable_scope("fm_layer", reuse=tf.AUTO_REUSE):
                self.FM_W = tf.get_variable(name='fm_w', shape=[self.feature_size, 1], initializer=tf.glorot_normal_initializer())
                self.FM_V = tf.get_variable(name='fm_v', shape=[self.feature_size, self.fm_v_size], initializer=tf.glorot_normal_initializer())
                self.FM_B = tf.Variable(tf.constant(0.0), dtype=tf.float32 ,name="fm_bias")  # W0
            print("%s: %s" % ("FM_W", self.FM_W))
            print("%s: %s" % ("FM_V", self.FM_V))
            print("%s: %s" % ("FM_B", self.FM_B))
            
            # ---------- w * x----------   
            Y_first = tf.reduce_sum(tf.multiply(self.FM_W, dense_vector), 2)  # None * F
            print("%s: %s" % ("Y_first", Y_first))
            
            # ---------- Vij * Vij* Xij ---------------
            embeddings = tf.multiply(self.FM_V, dense_vector) # None * V * X 
            # sum_square part
            summed_features_emb = tf.reduce_sum(embeddings, 1)  # sum(v*x)
            summed_features_emb_square = tf.square(summed_features_emb)  # (sum(v*x))^2

            # square_sum part
            squared_features_emb = tf.square(embeddings) # (v*x)^2
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # sum((v*x)^2)
            
            # second order
            Y_second = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # 0.5*((sum(v*x))^2 - sum((v*x)^2))            
            print("%s: %s" % ("Y_second", Y_second))
            
            # out = W * X + Vij * Vij* Xij
            FM_out_lay1 = tf.concat([Y_first, Y_second], axis=1) 
            Y_Out = tf.reduce_sum(FM_out_lay1, 1)
            # out = out + bias
            y_d = tf.reshape(Y_Out,shape=[-1])
            Y_bias = self.FM_B * tf.ones_like(y_d, dtype=tf.float32) # Y_bias
            Y_Out = tf.add(Y_Out, Y_bias, name='Y_Out') 
            print("%s: %s" % ("Y_bias", Y_bias))
            print("%s: %s" % ("Y_Out", Y_Out))
            # ---------- score ----------  
            score=tf.nn.sigmoid(Y_Out,name='score')
            score=tf.reshape(score, shape=[-1, 1])
            print("%s: %s" % ("score", score))
        
        """ 3 定义损失函数和AUC指标 """
        with tf.name_scope("loss"):
            # loss：Squared_error，Cross_entropy ,FTLR
            if reg_type == 'l1_reg':
                regularization = tf.contrib.layers.l1_regularizer(self.reg_param)(self.FM_W)
            elif reg_type == 'l2_reg':
                regularization = self.reg_param * tf.nn.l2_loss(self.FM_W)
            else:  
                regularization = self.reg_param * tf.nn.l2_loss(self.FM_W)    
            
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
            w_zero_ratio = tf.reduce_mean(tf.to_float(tf.abs(self.FM_W) <= 1.0e-5))
            w_avg = tf.reduce_mean(self.FM_W)
            v_zero_ratio = tf.reduce_mean(tf.to_float(tf.abs(self.FM_V) <= 1.0e-5))
            v_avg = tf.reduce_mean(self.FM_V)            
            
        """ 4 设定optimizer """
        with tf.name_scope("optimizer"):
            #------bulid optimizer------
            with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
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
            tf.summary.scalar("v_avg", v_avg)
            tf.summary.scalar("v_zero_ratio", v_zero_ratio)
            tf.summary.histogram("FM_W", self.FM_W)
            tf.summary.histogram("FM_V", self.FM_V)
            # 好几个summary，所以这里要merge_all
            summary_op = tf.summary.merge_all()
            
        """6 返回结果 """
        return Y_Out, score, regularization, loss, auc, train_step, w_zero_ratio, w_avg, v_zero_ratio, v_avg, labels, score, summary_op


# ## 3）模型训练测试

# In[4]:

# 测试数据
filenames = '/data/csv-all'
data_type='csv'
feature_size = 530
batch_size = 6000
num_epochs = 200
next_element = process_data(data_type, filenames, feature_size, batch_size, num_epochs)  
    
# 模型参数   
feature_size = 530
fm_v_size = 20
loss_fuc = 'Cross_entropy'
train_optimizer = 'Adam'
learning_rate = 0.01
reg_type = 'l2_reg'
reg_param = 0.000
log_path='/data/log/FM_Cross_entropy_L2_0_20180816_01'

# 开始训练
bea_model = FM(feature_size, fm_v_size, loss_fuc, train_optimizer, learning_rate, reg_type, reg_param)
Y_Out, score, regularization, loss, auc, train_step, w_zero_ratio, w_avg, v_zero_ratio, v_avg, labels, score, summary_op = bea_model.train(next_element)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
gpu_fraction = 0.6
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
