
# coding: utf-8

# ## 1）环境准备

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

import tensorflow as tf

print tf.__version__
print tf.__path__


# ## 2）数据准备

# In[9]:

# 定义输入样本格式
_CSV_COLUMNS = [
     'age', 'workclass', 'fnlwgt', 'education', 'education_num',
     'marital_status', 'occupation', 'relationship', 'race', 'gender',
     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
     'income_bracket'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]
_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

"""Builds a set of wide and deep feature columns."""
def build_model_columns():
    # 1. 特征处理，包括：连续特征、离散特征、转换特征、交叉特征等

    # 连续特征 （其中在Wide和Deep组件都会用到）
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    # 离散特征
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # 离散hash bucket特征
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000
    )

    # 特征Transformations
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    )

    # 2. 设定Wide层特征
    """
    Wide部分使用了规范化后的连续特征、离散特征、交叉特征
    """
    # 基本特征列
    base_columns = [
        # 全是离散特征
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    # 交叉特征列
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000
        )
    ]

    # wide特征列
    wide_columns = base_columns + crossed_columns

    # 3. 设定Deep层特征
    """
    Deep层主要针对离散特征进行处理，其中处理方式有：
    1. Sparse Features -> Embedding vector -> 串联(连续特征)，其中Embedding Values随机初始化。
    2. 另外一种处理离散特征的方法是：one-hot和multi-hot representation. 此方法适用于低维度特征，其中embedding是通用的做法
    其中：采用embedding_column(embedding)和indicator_column(multi-hot)API
    """
    # deep特征列
    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),

        # embedding特征
        tf.feature_column.embedding_column(occupation, dimension=8)
    ]
    return wide_columns, deep_columns

# Estimator Input
# 定义输入
def input_fn(data_file, num_epochs, shuffle, batch_size):
    """为Estimator创建一个input function"""
    assert tf.gfile.Exists(data_file), "{0} not found.".format(data_file)
    def parse_csv(line):
        print("Parsing", data_file)
        # tf.decode_csv会把csv文件转换成Tensor。其中record_defaults用于指明每一列的缺失值用什么填充。
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        # tf.equal(x, y) 返回一个bool类型Tensor， 表示x == y, element-wise
        return features, tf.equal(labels, '>50K') 
    dataset = tf.data.TextLineDataset(data_file).map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


# ## 3）模型准备

# In[10]:

# Wide & Deep Model
def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 50]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
          model_dir=model_dir,
          feature_columns=wide_columns,
          config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
          model_dir=model_dir,
          feature_columns=deep_columns,
          hidden_units=hidden_units,
          config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
          model_dir=model_dir,
          linear_feature_columns=wide_columns,
          dnn_feature_columns=deep_columns,
          dnn_hidden_units=hidden_units,
          config=run_config)
    
# 模型路径
model_type = 'widedeep'
model_dir = '/data/model/wide_deep'

# Wide & Deep 联合模型
model = build_estimator(model_dir, model_type)


# ## 4）模型训练

# In[11]:

# 训练参数
train_epochs = 10
batch_size = 5000
train_file = '/data/adult.data'
test_file = '/data/adult.test'

# 6. 开始训练
for n in range(train_epochs):
    # 模型训练
    model.train(input_fn=lambda: input_fn(train_file, train_epochs, True, batch_size))
    # 模型评估
    results = model.evaluate(input_fn=lambda: input_fn(test_file, 1, False, batch_size))
    # 打印评估结果
    print("Results at epoch {0}".format((n+1) * train_epochs))
    print('-'*30)
    for key in sorted(results):
        print("{0:20}: {1:.4f}".format(key, results[key]))
