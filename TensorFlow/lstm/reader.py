# -*- coding: utf-8 -*-
# 读取PTB文本集
#
# Author: Igor

import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace('\n', '<eos>').split()


def _build_vocab(filename):
    '''
    构建词典
    :param filename:
    :return:
    '''
    data = _read_words(filename)
    counter = collections.Counter(data)
    counter_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*counter_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data]


def ptb_raw_data(data_path=None):
    '''
    从指定的路径读取PTB文本,将字符转换为数字id
    执行输入的mini_batching

    :param data_path: simple-example.tgz 所在的路径
    :return: tuple(train_data,valid_data,test_data,vocabulary)
    '''

    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)  # 文档映射为id
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def iterator(raw_data, batch_size, num_steps):
    '''
    Iterate on the raw PTB data
    :param raw_data:one of the raw data outputs from ptb_raw_data
    :param batch_size:batch size
    :param num_steps:number of unrolls
    :yields:[batch_size,num_steps]
    '''

    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)  # 数据的长度
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0,decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


if __name__ == '__main__':
    path = 'data/'
    t, va_, te_, length = ptb_raw_data(path)
    # print(t)
    print(iterator(t,20,5))
