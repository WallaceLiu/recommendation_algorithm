# ## 2）数据准备Dataset格式
# 每一行解析，解析标签csv格式
# 5805 17357
def decode_csv(line):
    # 按照,分割，取label和feature
    columns = tf.string_split([line], ' ')
    center_words = tf.reshape(tf.string_to_number(columns.values[0], out_type=tf.int32),[-1])
    target_words = tf.reshape(tf.string_to_number(columns.values[1], out_type=tf.int32),[-1])
    return {'center_words': center_words, 'target_words': target_words}
# 文件读取，采用dataset格式
def read_my_file_format(filenames, batch_size, num_epochs=1):
    # 读取文件
    dataset = tf.data.TextLineDataset(filenames).map(lambda x: decode_csv(x)).prefetch(batch_size).cache()
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
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
def process_data(my_path, batch_size=32, num_epochs=1):
    filenames = get_file_list(my_path)
    next_element = read_my_file_format(filenames, batch_size, num_epochs)
    return next_element
    
filenames = "./windows_skip_sample.csv"
batch_size = 1000
num_epochs = 200
next_element = process_data(filenames, batch_size, num_epochs)