
# coding: utf-8

# In[ ]:


import xlearn as xl

# 1 模型建立
# model = create_linear()  #  Create linear model
# model = create_fm()      #  Create factorization machines
# model = create_ffm()     #  Create field-aware factorizarion machines.
ffm_model = xl.create_ffm() # 建立field-aware factorization machine模型
ffm_model.setTrain("./small_train.txt")  # 设置训练数据
ffm_model.setValidate("./small_test.txt")  # 设置测试数据

# 2 模型参数:
# task: {'binary',  # 二元分类
#      'reg'}     # 回归
# metric: {'acc', 'prec', 'recall', 'f1', 'auc', # 分类指标
#       'mae', 'mape', 'rmse', 'rmsd'}  # 回归指标
# lr: float value  # 学习率
# lambda: float value  #正则因子
# 其它超参因子参照API说明：https://xlearn-doc-cn.readthedocs.io/en/latest/all_api/index.html
param = {'task':'binary', 'lr':0.1, 'lambda':0.001, 'metric':'auc'}

# 3 训练模型
# The trained model will be stored in ffm_model.out
ffm_model.fit(param, './ffm_model.out')

# 4 测试
ffm_model.setTest("./small_test.txt")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# 预测结果
# The output result will be stored in output.txt
ffm_model.predict("./ffm_model.out", "./output.txt")

