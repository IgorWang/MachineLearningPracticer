__author__ = 'igor'

import tensorflow.python.platform

import collections
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
import urllib.request

# 步骤1:下载数据
url = 'http://mattmahoney.net/dc/'


def mayby_download(filename, expected_bytes):
    """
    下载数据(如果文件不存在),且保证数据是正确的大小
    :param filename:
    :param expected_bytes:
    :return:
    """
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)  # 返回文件的属性
    if statinfo.st_size == expected_bytes:
        print("Found and verified", filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Faied to verify ' + filename + '. Can you get to it with a browser?'
        )
    return filename


# 读取数据至字符串
def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return f.read(name).split()
    f.close()


# 步骤二:构建字典并把稀有的词替换为UNKtoken
vocabulary_size = 50000


def build_dataset(words):
    '''

    :param words:
    :return:data-值是单词的索引,用索引替换了单词,data就是出现的顺序
            count-[['UNK',-1],['单词',频次]] 包含 vocabulary_size + 1个
            dictionary-{'单词':索引}
    '''
    count = [['UNK', -1]]
    # words 是词表
    count.extend(collections.Counter(words).most_common(vocabulary_size))  # 筛选出频次最高的vocabulary_size个词
    dictionary = dict()  # 词:索引
    for word, _ in count:
        dictionary[word] = len(dictionary)  # 每个词和对应的索引
    data = list()  # 存储单词的索引
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:  # 如果单词没有在字典中,则索引到UNK
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # 索引:词
    return data, count, dictionary, reverse_dictionary


data_index = 0


# 步骤三: 为skip-gram生成训练batch
# skip-gram中心词预测环境
# 随机梯度下降,每次只训练一批数据,加快训练的速度
def generate_batch(batch_size, num_skips, skip_window):
    '''

    :param batch_size:训练样本的个数
    :param num_skips: 样本的间隔
    :param skip_window: 上下文窗口大小
    :return:数据和对应的标签
    '''
    global data_index  # 全局变量,保证了下一次生成batch的连续性
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [skip_window target skip_window],上下文环境
    buffer = collections.deque(maxlen=span)
    for _ in range(span):  # buffer里存储了一个上下文长度的词
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)  # 防止下标越界
    for i in range(batch_size // num_skips):
        target = skip_window  # 目标
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[target]
        buffer.append(data[data_index])  # 输入下一个词
        data_index = (data_index + 1) % len(data)
    return batch, labels


if __name__ == '__main__':
    filename = mayby_download('text8.zip', 31344016)
    words = read_data(filename)
    print('Data size', len(words))
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    del words  # 节省内存
    print('Most common words(+UNK)', count[:5])
    print('Sample data', data[:10])
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], '->', labels[i, 0])
        print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])
