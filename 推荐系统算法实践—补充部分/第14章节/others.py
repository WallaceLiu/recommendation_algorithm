import numpy as np
import tensorflow as tf
import pandas as pd
import random
import math
import re
from os import path, listdir
from tensorflow.contrib import layers
from sklearn import metrics
import time
import datetime

# ## 1）数据准备Dataset格式
# 每一行解析,格式:0229|0,0,0,0,0,0,0,0,0,0,0,0,0,0|1,1173,0,0,0|18578
def decode_sequence(line, continuous_size, item_size):
    columns = tf.string_split([line], '|')
    normalized_continuous_features = tf.string_to_number(tf.string_split([columns.values[1]], ',').values[0:continuous_size], out_type=tf.int32, name = "normalized_continuous_features")
    hist_click = tf.string_to_number(tf.string_split([columns.values[2]], ',').values[0:item_size], out_type=tf.int32, name = "hist_click")
    label = tf.reshape(tf.string_to_number(columns.values[3], out_type=tf.float32, name = "label"), [-1])
    return {"label": label, "hist_click":  hist_click, "normalized_continuous_features": normalized_continuous_features}

# 文件读取，采用dataset格式
def read_my_file_format(data_type, filenames, continuous_size, item_size, batch_size, num_epochs=1):
    # 读取文件
    print filenames
    dataset = tf.data.TextLineDataset(filenames).map(lambda x: decode_sequence(x, continuous_size, item_size)).prefetch(batch_size).cache()        
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

# 文件列表
def get_file_list(my_path):
    files = []    
    if path.isdir(my_path):
        [files.append(path.join(my_path, p)) for p in listdir(my_path) if path.isfile(path.join(my_path, p))]
    else:
        files.append(my_path)
    return files

# 数据处理
def process_data(data_type, my_path, continuous_size, item_size, batch_size=32, num_epochs=1):
    filenames = get_file_list(my_path)
    next_element = read_my_file_format(data_type, filenames, continuous_size, item_size, batch_size, num_epochs)
    return next_element

# In[15]:

# 测试数据
filenames = '/data/001'
continuous_size = 16
item_size = 5
batch_size = 3
num_epochs = 1
data_type = 'sequence'
next_element = process_data(data_type, filenames, continuous_size, item_size, batch_size, num_epochs)

# ## 2）定义YouTubeNet模型            
        """ 6 多层感知器神经网络计算，最终得到用户的embedding向量U：[batch_size, embedding_size] """  
        print("6 多层感知器神经网络计算")
        with tf.name_scope('MLP'):
            with tf.variable_scope("MLP", reuse=tf.AUTO_REUSE):
                # 第一层：（embedding_size + normalized_continuous_features_length） * embedding_size
                # 第二层： embedding_size * embedding_size
                weights = {
                    'h1': tf.Variable(tf.random_normal([self.embedding_size + self.normalized_continuous_features_length, self.embedding_size])),
                    'h2': tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size]))
                }
                biases = {
                    'b1': tf.Variable(tf.random_normal([self.embedding_size])),
                    'out': tf.Variable(tf.random_normal([self.embedding_size]))
                }
                print("%s: %s" % ("weights", weights))
                print("%s: %s" % ("biases", biases))
                layer_1 = tf.add(tf.matmul(all_concat, weights['h1']), biases['b1'])
                layer_1 = tf.nn.relu(layer_1)
                print("%s: %s" % ("layer_1", layer_1))
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['out'])
                print("%s: %s" % ("layer_2", layer_2))
                layer_out = tf.nn.relu(layer_2)
                print("%s: %s" % ("layer_out", layer_out))

        """ 7 Softmax计算，用户的embedding向量U 乘以 物品的embedding向量V，然后通过Softmax计算结果，其中Loss采用NCE负采样方法 """       
        print("7 最后一层Softmax计算")
        with tf.name_scope('Softmax_Classifer'):
            with tf.variable_scope("softmax_classifer", reuse=tf.AUTO_REUSE):            
                # NCE LOSS
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=self.weights,
                        biases=self.biases,
                        labels=label,
                        inputs=layer_out,
                        num_sampled=self.num_sampled,
                        num_classes=self.item_count
                    )
                )
                print("%s: %s" % ("loss", loss))
                # LOSS优化方法
                train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(loss)
                # Softmax的预测结果
                out = tf.nn.softmax(tf.matmul(layer_out, tf.transpose(self.weights)) + self.biases, dim=1)
                print("%s: %s" % ("out", out))