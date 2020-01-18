
# coding: utf-8

# ## 1）环境准备

# In[1]:

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import math
import re
from os import path, listdir
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print tf.__version__
print tf.__path__


# ## 2）数据准备Dataset格式

# In[2]:

# 每一行解析，解析标签csv格式
# 5805 17357
# 数据处理
def process_data(my_path, batch_size=32, num_epochs=1):
    filenames = get_file_list(my_path)
    next_element = read_my_file_format(filenames, batch_size, num_epochs)
    return next_element
# 创建session，指定GPU或者CPU使用率
def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# ## 3）Skip-gram模型

# In[3]:

class SkipGram(object):
    """ 初始化成员变量 """
    def __init__(self, vocab_size, embed_size, num_sampled, train_optimizer, learning_rate):
        # 字典长度
        self.vocab_size = vocab_size
        # 词向量长度
        self.embed_size = embed_size
        # 负采样数量
        self.num_sampled = num_sampled
        # 优化方法
        self.train_optimizer = train_optimizer
        # 学习率
        self.learning_rate = learning_rate
        # aglobal_step
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def train(self, batch_data):
        """ 1 定义输入数据 """
        with tf.name_scope('input_data'):
            # center_words
            center_words = tf.reshape(batch_data['center_words'], shape=[-1])            
            # target_words
            target_words = tf.reshape(batch_data['target_words'], shape=[-1,1])
            print("%s: %s" % ("center_words", center_words))
            print("%s: %s" % ("target_words", target_words))
            
        """ 2 定义网络输出 """
        with tf.name_scope("Comput_Score"):
            # 词向量矩阵
            with tf.variable_scope("embed", reuse=tf.AUTO_REUSE):
                self.embedding_dict = tf.get_variable(name='embed', shape=[self.vocab_size, self.embed_size], initializer=tf.glorot_uniform_initializer())
            print("%s: %s" % ("embedding_dict", self.embedding_dict))
            
            # 模型内部参数矩阵
            with tf.variable_scope("nce", reuse=tf.AUTO_REUSE): 
                self.nce_weight = tf.get_variable(name='nce_weight', shape=[self.vocab_size, self.embed_size], initializer=tf.glorot_normal_initializer())
                self.nce_biases = tf.get_variable(name='nce_biases', shape=[1], initializer=tf.constant_initializer(0.0))         
                print("%s: %s" % ("nce_weight", self.nce_weight))
                print("%s: %s" % ("nce_biases", self.nce_biases))
                
            # 将输入序列向量化
            # 其实就是一个简单的查表
            embed = tf.nn.embedding_lookup(self.embedding_dict, center_words, name='embed')
            print("%s: %s" % ("embed", embed))
            
            # 得到NCE损失(负采样得到的损失)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights = self.nce_weight,  # 权重
                    biases = self.nce_biases,   # 偏差
                    labels = target_words, # 输入的标签
                    inputs = embed,             # 输入向量
                    num_sampled = self.num_sampled, # 负采样的个数
                    num_classes = self.vocab_size # 字典数目
                )
            ) 
            print("%s: %s" % ("loss", loss))
            
        """ 3 设定optimizer """
        with tf.name_scope("optimizer"):
            with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
                #------bulid optimizer------
                if train_optimizer == 'Adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
                elif train_optimizer == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
                train_step = optimizer.minimize(loss, global_step=self.global_step)               

        """4 设定summary，以便在Tensorboard里进行可视化 """
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            tf.summary.histogram("embedding_dict", self.embedding_dict)
            # 好几个summary，所以这里要merge_all
            summary_op = tf.summary.merge_all()
            
        """5 返回结果 """
        return train_step, loss, summary_op


# ## 4）模型训练测试

# In[4]:

# 测试数据
filenames = "/data/windows_skip_sample.csv"
batch_size = 100000
num_epochs = 200
next_element = process_data(filenames, batch_size, num_epochs)
    
# 模型参数   
vocab_size = 6834
embed_size = 30
num_sampled = 50
train_optimizer = 'Adam'
learning_rate = 0.01
log_path='/data/log/20180915'

# 开始训练
bea_model = SkipGram(vocab_size, embed_size, num_sampled, train_optimizer, learning_rate)
train_step, loss, summary_op = bea_model.train(next_element)

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
            a, b, summary = sess.run([train_step, loss, summary_op])
            if batch_cnt % 1000 == 0 or batch_cnt <= 10:
                print("batch: {} loss: {:.4f}".format(batch_cnt, b))
                writer.add_summary(summary, batch_cnt)
    except tf.errors.OutOfRangeError:
        print("Train end of dataset")   

