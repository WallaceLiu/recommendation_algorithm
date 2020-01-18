
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

import time
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print tf.__version__
print tf.__path__


# ## 2）数据准备Dataset格式

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


# ## 3）Debug代码

# In[3]:

""" 0 测试数据 """
filenames = '/data/data0/001'
feature_size = 530
fm_v_size = 10
batch_size = 3
num_epochs = 1
data_type = 'libsvm'
batch_data = process_data(data_type, filenames, feature_size, batch_size, num_epochs)
print("%s: %s" % ("batch_data", batch_data))


# In[9]:

""" 1 定义输入数据 """
# 标签：[batch_size, 1]
labels = batch_data['labels']
# 用户特征向量：[batch_size, feature_size]
dense_vector = tf.reshape(batch_data['dense_vector'], shape=[-1, feature_size, 1]) # None * feature_size * 1
print("%s: %s" % ("dense_vector", dense_vector))
print("%s: %s" % ("labels", labels))


# In[10]:

""" 2 定义网络输出 """
# FM参数，生成或者获取W V
with tf.variable_scope("lr_layer", reuse=tf.AUTO_REUSE):
    FM_W = tf.get_variable(name='fm_w', shape=[feature_size, 1], initializer=tf.glorot_normal_initializer())
    FM_V = tf.get_variable(name='fm_v', shape=[feature_size, fm_v_size], initializer=tf.glorot_normal_initializer())
    FM_B = tf.Variable(tf.constant(0.0), dtype=tf.float32 ,name="fm_bias")  # W0
print("%s: %s" % ("FM_W", FM_W))
print("%s: %s" % ("FM_V", FM_V))
print("%s: %s" % ("FM_B", FM_B))


# In[11]:

# ---------- w * x ----------   
Y_first = tf.reduce_sum(tf.multiply(FM_W, dense_vector), 2)  # None * F
print("%s: %s" % ("Y_first", Y_first))


# In[12]:

# ---------- Vij * Vij* Xij ---------------
embeddings = tf.multiply(FM_V, dense_vector) # None * V * X 
print("%s: %s" % ("embeddings", embeddings))
# sum_square part
summed_features_emb = tf.reduce_sum(embeddings, 1)  # sum(v*x)
summed_features_emb_square = tf.square(summed_features_emb)  # (sum(v*x))^2

# square_sum part
squared_features_emb = tf.square(embeddings) # (v*x)^2
squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # sum((v*x)^2)
            
# second order
Y_second = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # 0.5*((sum(v*x))^2 - sum((v*x)^2))            
print("%s: %s" % ("Y_second", Y_second))


# In[15]:

# out = W * X + Vij * Vij* Xij
FM_out_lay1 = tf.concat([Y_first, Y_second], axis=1) 
print("%s: %s" % ("FM_out_lay1", FM_out_lay1))

Y_Out = tf.reduce_sum(FM_out_lay1, 1)
print("%s: %s" % ("Y_Out", Y_Out))


# In[16]:

# out = out + bias
y_d = tf.reshape(Y_Out,shape=[-1])
Y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32) # Y_bias
Y_Out = tf.add(Y_Out, Y_bias, name='Y_Out') 
print("%s: %s" % ("Y_bias", Y_bias))
print("%s: %s" % ("Y_Out", Y_Out))


# In[17]:

# ---------- score ----------  
score=tf.nn.sigmoid(Y_Out,name='score')
score=tf.reshape(score, shape=[-1, 1])
print("%s: %s" % ("score", score))


# In[18]:

""" 3 定义损失函数和AUC指标 """
reg_type = 'l2_reg'
loss_fuc = 'Cross_entropy'
reg_param = 0.01
learning_rate = 0.01
print("%s: %s" % ("reg_type", reg_type))
print("%s: %s" % ("loss_fuc", loss_fuc))
print("%s: %s" % ("reg_param", reg_param))
print("%s: %s" % ("learning_rate", learning_rate))


# In[19]:

# loss：Squared_error，Cross_entropy ,FTLR
if reg_type == 'l1_reg':
    regularization = reg_param * tf.reduce_sum(tf.abs(FM_W))
elif reg_type == 'l2_reg':
    regularization = reg_param * tf.nn.l2_loss(FM_W) 
else:  
    regularization = reg_param * tf.nn.l2_loss(FM_W)                 
            
if loss_fuc == 'Squared_error':
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
elif loss_fuc == 'Cross_entropy':
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(Y_Out, [-1]), labels=tf.reshape(labels, [-1]))) + regularization
elif loss_fuc == 'FTLR':
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization


# In[20]:

# AUC                  
auc = tf.metrics.auc(labels, score)
print("%s: %s" % ("labels", labels))
print("%s: %s" % ("score", score))


# In[21]:

# w为0的比例,w的平均值
w_zero_ratio = tf.reduce_mean(tf.to_float(tf.abs(FM_W) <= 1.0e-5))
w_avg = tf.reduce_mean(FM_W)
v_zero_ratio = tf.reduce_mean(tf.to_float(tf.abs(FM_V) <= 1.0e-5))
v_avg = tf.reduce_mean(FM_V)  


# In[22]:

""" 4 设定optimizer """
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_step = optimizer.minimize(loss, global_step=global_step)


# In[23]:

""" 分步调试，对上面各个步骤中的变量值进行打印和查看，以方便定位问题 """
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
with tf.device('/cpu:0'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init_op)
    a, b = sess.run([Y_Out, score])
    print a
    print b
