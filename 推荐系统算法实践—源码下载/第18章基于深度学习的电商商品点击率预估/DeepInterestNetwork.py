
# coding: utf-8

# ## 1）环境准备

# In[28]:


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

print(tf.__version__)
print(tf.__path__)


# ## 2）数据准备

# In[23]:


# 获取商品和类目的embedding数据
def get_embedding():
    # 类目embedding数据
    # 商品embedding数据
    return {"category_list": category_list, "golds_list": golds_list}

# 读取用户行为数据，格式：点击|浏览序列|点击序列|购买序列|类目兴趣序列|用户画像特征
# 1000000123|0,0,0,0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,0,0||0,0,0,0,0,0,0,0,0,0|1,1173,0,0,0
def decode_sequence(line, gold_size, category_size, profile_size):
	   # 数据解析	
    return {"label": label, "goods": goods, "category": category, "profile": profile}

# 数据处理
def process_data(my_path, gold_size, category_size, other_size, batch_size=32, num_epochs=1):
    filenames = get_file_list(my_path)
    next_element = read_my_file_format(filenames, gold_size, category_size, other_size, batch_size, num_epochs)
    return next_element


# In[29]:


# 测试数据
filenames = 'D:\\Data\\GoldsData\\User\\user_data.csv'
batch_size = 2
num_epochs = 1
gold_size = 10
category_size = 8
other_size = 12
next_element = process_data(filenames, gold_size, category_size, other_size, batch_size, num_epochs)
print(next_element['label'])
print(next_element['goods'])
print(next_element['category'])
print(next_element['profile'])

my_device='/cpu:0'
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
with tf.device(my_device):
    sess = tf.Session()
    sess.run(init_op)
    label, goods, category, other = sess.run([next_element['label'],next_element['goods'],next_element['category'],next_element['profile']])
    print(label)
    print(goods)
    print(category)
    print(other)
    


# In[30]:


# embedding数据
embedding = get_embedding()
print(embedding['category_list'].shape)
print(embedding['golds_list'].shape)
print(embedding['golds_list'])
print(embedding['golds_list'])


# ## 3）定义DeepInterestNetwork模型

# In[31]:


class DeepInterestNetwork(object):
    """ 一、初始化成员变量 """
    def __init__(self, 
                 goods_size, 
                 goods_embedding_size, 
                 category_embedding_size, 
                 num_sampled, 
                 learning_rate, 
                 attention_size, 
                 goods_input_length, 
                 category_input_length, 
                 profile_input_length, 
                 log_path):
        # 商品池大小
        self.goods_size = goods_size
        # 商品embedding大小
        self.goods_embedding_size = goods_embedding_size
        self.category_embedding_size = category_embedding_size
        # NCE采样数量
        self.num_sampled = num_sampled
        # 学习率
        self.learning_rate = learning_rate
        # attention层大小
        self.attention_size = attention_size
        # 用户购买序列特征长度
        self.goods_input_length = goods_input_length
        # 用户类目兴趣特征长度
        self.category_input_length = category_input_length
        # 用户画像特征长度
        self.profile_input_length = profile_input_length
        # log_path
        self.log_path = log_path        
        
    """ 二、计算网络最后一层的输出 """ 
    def _comput_lay_out(self, batch_data):  
        """ 1 定义输入数据 """
        print("1 定义输入数据" )
        with tf.name_scope('input_data'):
            # 用户画像特征向量：[batch_size, profile_input_length]
            input_profile = batch_data['profile']
            # 用户类目特征向量：[batch_size, category_input_length]
            input_category = batch_data['category']
            # 用户购买序列特征向量：[batch_size, goods_input_length]
            input_goods = batch_data['goods']
            print("%s: %s" % ("input_profile", input_profile))
            print("%s: %s" % ("input_goods", input_goods))    
            print("%s: %s" % ("input_category", input_category))
            
            # 计算gold序列中0的比例
            batch_goods_ratio = tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.abs(input_goods) > 0),1),0)
            
        """ 2 对用户行为序列进行embedding_lookup查找，得到用户的行为embed向量 """    
        # 省略，可以参照第14章。
            
        """ 3 attention机制，根据用户行为embed向量，通过多层感知神经网络，最后通过Saftmax得到alpha权重向量 """   
        print("3 对用户序列进行attention层计算" )
        with tf.name_scope('attention_layer'):
            with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
                # 全连接层计算
                # inputs shape: [batch_size, goods_input_length, embedding_size]
                # h: [batch_size, goods_input_length, embedding_size]
                h = layers.fully_connected(inputs_goods_emb, self.attention_size, activation_fn=tf.nn.tanh)
                print("%s: %s" % ("h", h))
            
                # 输出层计算
                # u_context: importance vector
                u_context = tf.Variable(tf.truncated_normal([self.attention_size]))
                hu_sum = tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True)  
                print("%s: %s" % ("hu_sum", hu_sum))
                # 防止 exp 溢出
                hu_max = tf.reduce_max(hu_sum, axis=1, keep_dims=True)
                print("%s: %s" % ("hu_max", hu_max))
                hu_normal = hu_sum - hu_max
                print("%s: %s" % ("hu_normal", hu_normal))
            
                # Softmax计算
                # hu_sum: [batch_size, goods_input_length, 1]
                exp = tf.exp(hu_normal)
                exp_adapt = exp
                print("%s: %s" % ("exp_adapt", exp_adapt))
    
                exp_adapt_sum = tf.reduce_sum(exp_adapt, axis=1, keep_dims=True)
                print("%s: %s" % ("exp_adapt_sum", exp_adapt_sum))
                alpha = tf.div(exp_adapt, exp_adapt_sum)
                print("%s: %s" % ("alpha", alpha))

                # attention计算，[batch_size, embedding_size]
                atten_embed = tf.reduce_sum(tf.multiply(inputs_goods_emb, alpha), axis=1) 
                print("%s: %s" % ("atten_embed", atten_embed))         
            
        """ 4 用户特征向量拼接 """   
        # 省略，可以参照第14章。
                    
        """ 5 多层感知器神经网络计算，最终得到用户的embedding向量U：[batch_size, embedding_size] """  
        # 省略，可以参照第14章。

    
    """ 三、网络训练 """ 
    def train(self, batch_data, goods_embedding, category_embedding):            
        """ 1 Embedding初始化 """
        with tf.name_scope('embedding'):
            self.goods_embedding = tf.convert_to_tensor(goods_embedding, dtype=tf.float32)
            self.category_embedding = tf.convert_to_tensor(category_embedding, dtype=tf.float32)
            print("%s: %s" % ("goods_embedding", self.goods_embedding))
            print("%s: %s" % ("category_embedding", self.category_embedding))
            with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                self.nce_biases = tf.get_variable(name='nce_biases', shape=[self.goods_size], initializer=tf.constant_initializer(0.0))         
                print("%s: %s" % ("nce_biases", self.nce_biases))   
                
        """ 2 计算深度神经网络的最后一层输出 """
        layer_out, batch_goods_ratio = self._comput_lay_out(batch_data)  
        # 用户标签：[batch_size, 1]
        input_label = batch_data['label']
        print("%s: %s" % ("input_label", input_label))

        """ 3 Softmax计算，用户的embedding向量U乘以商品的embedding向量V，然后通过Softmax计算结果，其中Loss采用NCE负采样方法 """       
        print("3 最后一层Softmax计算")
        # 省略，可以参照第14章。
                
        """4 设定summary，以便在Tensorboard里进行可视化 """
        print("4 设定summary" ) 
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            tf.summary.histogram("nce_biases", self.nce_biases)
            # 好几个summary，所以这里要merge_all
            summary_op = tf.summary.merge_all()   
            
        """5 返回结果 """
        return out, loss, batch_goods_ratio, input_label, summary_op, train_step    
    
    """ 四、预测计算 """ 
    def predict(self, batch_data):      
        """ 1 计算深度神经网络的最后一层输出 """
        layer_out, _ = self._comput_lay_out(batch_data)   
        
        """ 2 计算Softmax的预测结果 """
        predict_score = tf.nn.softmax(tf.matmul(layer_out, tf.transpose(self.goods_embedding)) + self.nce_biases, dim=1)
        # 结果返回
        return predict_score


# ## 4）模型训练测试

# In[32]:


# 数据参数
print("0 数据准备和参数设置" ) 
filenames = 'D:\\Data\\GoldsData\\User\\user_data.csv'
batch_size = 2000
num_epochs = 1000
gold_size = 10
category_size = 8
profile_size = 12
next_element = process_data(filenames, gold_size, category_size, other_size, batch_size, num_epochs)
print("%s: %s" % ("next_element", next_element))

# 模型参数
goods_size = 40742
goods_embedding_size = 100
category_embedding_size = 10
num_sampled = 32
learning_rate = 0.01
attention_size = 60
goods_input_length = gold_size * 3
category_input_length = category_size
profile_input_length = profile_size
log_path='D:\\Data\\log\\20180915'

# embedding参数
embedding = get_embedding()
goods_embedding = embedding['golds_list']
category_embedding = embedding['category_list']
print("%s: %s" % ("goods_embedding.shape",  goods_embedding.shape))
print("%s: %s" % ("category_embedding.shape",  category_embedding.shape))

# 开始训练
golds_rec_model = DeepInterestNetwork(goods_size, 
                  goods_embedding_size, 
                  category_embedding_size,
                  num_sampled, 
                  learning_rate, 
                  attention_size,  
                  goods_input_length, 
                  category_input_length, 
                  profile_input_length, 
                  log_path)
out, loss, batch_goods_ratio, input_label, summary_op, train_step = golds_rec_model.train(next_element, goods_embedding, category_embedding)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
my_device='/cpu:0'
with tf.device(my_device):
    sess = tf.Session()
    sess.run(init_op)
    batch_cnt = 0
    #选定可视化存储目录
    writer = tf.summary.FileWriter(log_path, sess.graph)
    print("5 迭代过程" ) 
    try:
         while True:
            batch_cnt = batch_cnt + 1
            a, b, c, d, summary, _ = sess.run([out, loss, batch_goods_ratio, input_label, summary_op, train_step])
            if batch_cnt % 200 == 0 or batch_cnt <= 10:
                print("batch: {}    loss: {}    gold_ratio: {}".format(batch_cnt, b, c))
                writer.add_summary(summary, batch_cnt)
    except tf.errors.OutOfRangeError:
        print("Train end of dataset")     

