
# coding: utf-8

# ## 0）环境准备

# In[8]:

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

# In[14]:

# 每一行解析 sequence格式
# 351702070890229|0,0,0,0,0,0,0,0,0,0,0,0,0,0|1,1173,0,0,0|18578

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


# In[15]:

# 测试数据
filenames = '/data/sequence_normalize/001'
item_size = 5
batch_size = 3
num_epochs = 1
data_type = 'sequence'
next_element = process_data(data_type, filenames, item_size, batch_size, num_epochs)
# print next_element['label']
# print next_element['hist_click']
# print next_element['normalized_continuous_features']

gpu_fraction = 0.2
my_device='/gpu:0'
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
with tf.device(my_device):
    sess = get_session(gpu_fraction)
    sess.run(init_op)
    label, item, other = sess.run([next_element['label'],next_element['hist_click'],next_element['normalized_continuous_features']])
    print label
    print item
    print other


# ## 2）定义YouTubeNet模型

# In[26]:

class YouTubeNet(object):
    """ 初始化成员变量 """
    def __init__(self, 
                 item_count, 
                 embedding_size, 
                 num_sampled, 
                 learning_rate, 
                 hist_click_length, 
                 normalized_continuous_features_length, 
                 log_path):
        # 资源池大小
        self.item_count = item_count
        # embedding大小
        self.embedding_size = embedding_size
        # NCE采样数量
        self.num_sampled = num_sampled
        # 学习率
        self.learning_rate = learning_rate
        # 用户行为序列特征长度
        self.hist_click_length = hist_click_length
        # 用户其它特征长度
        self.normalized_continuous_features_length = normalized_continuous_features_length
        # log_path
        self.log_path = log_path

    def train(self, batch_data):
        """ 1 定义输入数据 """
        print("1 定义输入数据" )
        with tf.name_scope('input_data'):
            # 用户其它特征向量：[batch_size, normalized_continuous_features_length]
            normalized_continuous_features = batch_data['normalized_continuous_features']
            # 用户行为序列特征向量：[batch_size, hist_click_length]
            hist_click = batch_data['hist_click']
            # 用户标签：[batch_size, 1]
            label = batch_data['label']
            # 计算item序列中0的比例
            batch_item_ratio = tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.abs(hist_click) > 0),1),0)
            print("%s: %s" % ("normalized_continuous_features", normalized_continuous_features))
            print("%s: %s" % ("hist_click", hist_click))    
            print("%s: %s" % ("label", label))
            
        """ 2 Embedding初始化 """
        # 初始化物品embedding向量V：[item_count, embedding_size]
        print("2 Embedding初始化" )
        with tf.name_scope('embedding'):
            with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                self.weights = tf.Variable(tf.truncated_normal([self.item_count, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size)))
                self.biases = tf.Variable(tf.zeros([self.item_count]))
                print("%s: %s" % ("weights", self.weights))
                print("%s: %s" % ("biases", self.biases))
                
        """ 3 对用户行为序列进行embedding_lookup查找，得到用户的行为embed向量 """    
        print("3 对用户item序列进行embedding_lookup查找" )
        with tf.name_scope("embedding_lookup"):
            # weights：[item_count, embedding_size]
            # hist_click：[batch_size, hist_click_length]
            # embed：[batch_size, hist_click_length, embedding_size]
            inputs = tf.nn.embedding_lookup(self.weights, hist_click)
            print("%s: %s" % ("inputs", inputs))
            
        """ 4 pooling操作，根据用户行为embed向量，进行求和或者平均操作 """   
        print("4 对用户序列进行pooling操作" )
        with tf.name_scope('pooling_layer'):
            pooling_embed = tf.reduce_sum(inputs, axis=1) 
            print("%s: %s" % ("pooling_embed", pooling_embed))            
            
        """ 5 用户特征向量拼接 """   
        print("5 用户特征向量拼接")
        with tf.name_scope("all_concat"):
            all_concat = tf.concat([pooling_embed, normalized_continuous_features], 1)
            print("%s: %s" % ("all_concat", all_concat))
                        
        """ 6 多层感知器神经网络计算，最终得到用户的embedding向量U：[batch_size, embedding_size] """  
        # 省略，可以参照第13章或者第12章。

        """ 7 Softmax计算，用户的embedding向量U 乘以物品的embedding向量V，然后通过Softmax计算结果，其中Loss采用NCE负采样方法 """
        print("7 最后一层Softmax计算")
        with tf.name_scope('Softmax_Classifer'):
            with tf.variable_scope("softmax_classifer", reuse=tf.AUTO_REUSE):            
        # 省略，可以参照https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow/blob/master/youtube_recommendation.py。
                
        """8 设定summary，以便在Tensorboard里进行可视化 """
        print("8 设定summary" ) 
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            tf.summary.histogram("weightsweight", self.weights)
            # 好几个summary，所以这里要merge_all
            summary_op = tf.summary.merge_all()   
            
        """9 返回结果 """
        return out, loss, batch_item_ratio, label, summary_op, train_step


# ## 3）模型训练测试

# In[ ]:

# 数据参数
print("0 数据准备和参数设置" ) 
batch_size=2000
item_size = 30
num_epochs=1
filenames = '/data/001'
data_type = 'sequence'
next_element = process_data(data_type, filenames, item_size, batch_size, num_epochs)
print("%s: %s" % ("next_element", next_element))

# 模型参数
item_count = 99974
embedding_size = 32
num_sampled = 32
learning_rate = 0.01
hist_click_length = item_size * 3
f_size = hist_click_length + 2
normalized_continuous_features_length = f_size - hist_click_length - 1
log_path='/data/log/youtubenet_20180810_001'

# 开始训练
bea_model = YouTubeNet(item_count, embedding_size, num_sampled, learning_rate, hist_click_length, normalized_continuous_features_length, log_path)
out, loss, batch_item_ratio, label, summary_op, train_step = bea_model.train(next_element)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
gpu_fraction = 0.5
my_device='/cpu:0'
with tf.device(my_device):
    sess = get_session(gpu_fraction)
    sess.run(init_op)
    batch_cnt = 0
    #选定可视化存储目录
    writer = tf.summary.FileWriter(log_path, sess.graph)
    print("9 迭代过程" ) 
    try:
         while True:
            batch_cnt = batch_cnt + 1
            a, b, c, d, summary, _ = sess.run([out, loss, batch_item_ratio, label, summary_op, train_step])
            if batch_cnt % 400 == 0 or batch_cnt <= 10:
                print("batch: {}    loss: {}    item_ratio: {}".format(batch_cnt, b, c))
                writer.add_summary(summary, batch_cnt)
    except tf.errors.OutOfRangeError:
        print("Train end of dataset")     
