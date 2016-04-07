# -*- coding: utf-8 -*-
#  
#
# Author: Igor

import os

import tensorflow as tf

# 图像像素大小,更改该参数够整个模型架构都会改变
IMAGE_SIZE = 24

# 描述CIFAR-10数据集的全局常量
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    '''
    从数据中读取和解析样本数据

    :param filename_queue:文件名的字符串队列
    :return:
        一个单个样本的对象,包含的字段:
            height: 结果中行的数量 (32)
            width: 结果中列的数量 (32)
            depth: 颜色通道的数量 (3)
            key: 一个标量字符串Tensor描述该样本的filename(文件)&record(记录)
            label: 一个int32张量表示0-9范围内的标签
            uint8image:一个 [height,width,depth] uint8 Tensor(行,列,每个元素的[红;黄;绿])
    '''

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # 数据集中的数据是固定数量的格式
    record_bytes = label_bytes + image_bytes
    # 构建一个固定长度的读取器
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 返回样本的key和值
    result.key, value = reader.read(filename_queue)

    # 将一个字符串转换为一个uint8张量
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)


def distored_inputs(data_dir, batch_size):
    '''
    为CIFAR模型训练提供无序输入
    利用Reader ops
    :param data_dir:Path to the CIFAR-10 data directory
    :param batch_size:Number of images per batch
    :return:
        images:Images 4D tensor of [batch_size,IMAGE_SIZE,IMAGE_SIZE,3] size
        labels:Labels !D tensor of [batch_size] size
    '''

    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)

    # 根据文件的格式选择阅读器,构建队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 从阅读器中读取样本数据
    read_input = read_cifar10(filename_queue)
