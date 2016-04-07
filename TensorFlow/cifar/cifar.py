# -*- coding: utf-8 -*-
# cifar模型:图像分类
#
# Author: Igor

import gzip
import os
import re
import sys
import tarfile
import urllib.request

import tensorflow as tf

from TensorFlow.cifar import cifar10_input

FLAGS = tf.app.flags.FLAGS

# 基础的模型参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "number of images to process in a batch")
tf.app.flags.DEFINE_string('data_dir', 'data/',
                           "Path to the CIFAR-10 data directory")

# 描述CIFAR-10数据集的全局常量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASS = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# 训练过程的全局常量
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'




def maybe_download_and_extract():
    '''
    下载并提取数据集
    '''
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.mkdir(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


if __name__ == '__main__':
    pass