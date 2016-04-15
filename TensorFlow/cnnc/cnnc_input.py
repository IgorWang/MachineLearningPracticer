# -*- coding: utf-8 -*-
#  
#
# Author: Igor
import os

import tensorflow as tf

SEQUENCE_LENGTH = 59


def read_data(filename_queue):
    '''
    数据的读取
    :param filename_queue:
    :return:
    '''

    class DataRecord(object):
        pass

    result = DataRecord()

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1] * (SEQUENCE_LENGTH + 2)]
    record = tf.decode_csv(value,
                           record_defaults=record_defaults)
    features = tf.pack(record[0:SEQUENCE_LENGTH])
    labels = tf.pack(record[SEQUENCE_LENGTH:])


def distorted_inputs(filenames, batch_size):
    '''
    构建文本输入
    :param data_dir: 数据目录
    :param batch_size: 批量大小
    :return:
        document: 2D tensor of [batch_size,SEQUENCE_LENGTH]
        labels: 2D tensor [batch_size,2]
    '''
    for f in filenames:
        if not os.path.exists(f):
            raise ValueError("Failed to find file:" + f)

    # 文件的读取队列
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_data(filename_queue)
