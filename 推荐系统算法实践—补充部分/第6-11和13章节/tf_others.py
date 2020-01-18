import numpy as np
import tensorflow as tf
import pandas as pd
import random
import math
import re

from os import path, listdir
from sklearn import metrics
from tensorflow.contrib import layers

import time
import datetime

# ## 2）数据准备Dataset格式

# In[6]:

"""
解析CSV格式，对输入的每一行样本，进行格式解析，返回labels和dense_vector格式数据
例如输入csv格式字符串： 0.0,0.6666666666666666,0.5,0.0,0.0,0.0,0.0,0.7272727272727273,0.42857142857142855
函数参数：
line：需要解析的字符串；
feature_size：特征长度；
函数返回：
返回字典，格式：{'labels': labels, 'dense_vector': dense_vector}
labels：样本的labels；
dense_vector：样本的特征向量；
"""
def decode_csv(line, feature_size):
    # 按照,分割，取label和feature
    columns = tf.string_split([line], ',')
    labels = tf.reshape(tf.string_to_number(columns.values[0], out_type=tf.float32),[-1])
    dense_vector = tf.reshape(tf.string_to_number(columns.values[1:feature_size + 1], out_type=tf.float32),[feature_size])
    return {'labels': labels, 'dense_vector': dense_vector}

"""
采用DataSet格式读取文件
函数参数：
data_type：文件格式；
filenames：文件路径；
batch_size：Batch大小；
feature_size：特征长度；
num_epochs：样本复制多少次；
函数返回：
返回DataSet
"""
def read_my_file_format(data_type, filenames, feature_size, batch_size, num_epochs=1):
    # 读取文件
    print filenames
    dataset = tf.data.TextLineDataset(filenames).map(lambda x: decode_csv(x, feature_size)).prefetch(batch_size).cache()
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
def process_data(data_type, my_path, feature_size, batch_size=32, num_epochs=1):
    filenames = get_file_list(my_path)
    next_element = read_my_file_format(data_type, filenames, feature_size, batch_size, num_epochs)
    return next_element

# 测试数据
filenames = '/data/csv-00000'
feature_size = 530
batch_size = 3
num_epochs = 1
data_type = 'csv'
next_element = process_data(data_type, filenames, feature_size, batch_size, num_epochs)
print next_element['dense_vector']
print next_element['labels']
