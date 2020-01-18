from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from os import path, listdir
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib  
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random
print("Python Version: %s"%(platform.python_version()))


# ## 2）数据准备
def process_data(data_path, feature_size, test_rat=0.3, random_seed=0, train_batch_size=5000, test_batch_size=5000):
    # 读取文件
    filenames = get_file_list(data_path)
    data = load_svmlight_files(filenames, n_features=feature_size, dtype=np.float32)   
    # 合并所有文件
    merger_x = data[0].toarray()
    merger_y = data[1].reshape(-1, 1)
    for i in range(2,len(data)):
        if i % 2 == 0:
            x = data[i].toarray()
            merger_x=np.vstack((merger_x, x))
        else:
            y = data[i].reshape(-1, 1)
            merger_y=np.vstack((merger_y, y))

    # 生成x y datafarme
    feature_col = range(1,(feature_size + 1))
    x_frame = pd.DataFrame(merger_x ,columns=feature_col)
    y_frame = pd.DataFrame(merger_y)
    # 数据归一化
    minmax_scala=preprocessing.MinMaxScaler(feature_range=(0,1))
    scalafeature=minmax_scala.fit_transform(x_frame)
    scalafeature_frame = pd.DataFrame(scalafeature ,columns=x_frame.columns) 
    # 训练样本,测试样本生成
    X_train, X_test, Y_train, Y_test = train_test_split(scalafeature_frame, y_frame, test_size=test_rat, random_state=random_seed)
    # batch生成
    all_train = pd.concat([Y_train, X_train], axis=1)
    all_test = pd.concat([Y_test, X_test], axis=1)
    xy_train = np.array(all_train).reshape(-1, feature_size + 1)
    xy_test = np.array(all_test).reshape(-1, feature_size + 1) 
    train_batch = split_batch(xy_train, train_batch_size)
    test_batch = split_batch(xy_test, test_batch_size)
    return {"train_batch": train_batch, "test_batch": test_batch, "X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}

# 按照batch_size大小将数据进行切分，返回Batch数据
def split_batch(xy_data, batch_size=5000):
    # 计算batch数量
    all_len=xy_data.shape[0]
    n=int(round(float(all_len)/batch_size))
    if n == 0:
        n = 1        
    data_batch=[]
    # 生成每个batch
    for i in range(n):
        k1=i*batch_size
        if i < n-1:
            k2=(i+1)*batch_size
        elif i == (n-1) and (i+1)*batch_size <= all_len:
            k2=all_len
        else:
            k2=(i+1)*batch_size
        batch=xy_data[k1:k2,:]
        data_batch.append(batch)   
    return data_batch

# 根据文件目录，获取文件路径，返回文件路径列表
def get_file_list(my_path):
    files = []    
    if path.isdir(my_path):
        [files.append(path.join(my_path, p)) for p in listdir(my_path) if path.isfile(path.join(my_path, p))]
    else:
        files.append(my_path)     
    return files
    
    
# 数据测试
data_path = '/data'
test_rat=0.4
random_seed=0
train_batch_size=2000
test_batch_size=2000
feature_size=530

# 获取样本数据
data = process_data(data_path, feature_size, test_rat, random_seed, train_batch_size, test_batch_size)