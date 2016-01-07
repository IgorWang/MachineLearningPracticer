__author__ = 'igor'

"""Trains and Evaluates the MNIST netword using a feed dictionary"""

import os.path
import time

import tensorflow.python.platform
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist


# 模型的基本参数
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.'
                                        'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'if true,uses fake data'
                                         'fpr unit testing.')


def placeholder_inputs(batch_size):
    '''
    生成placeholder variables来表示输入tensors
    :param
        batch_size:placeholders
    :return:
        images_placeholder
        labels_placeholder
    '''
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size))
    return images_placeholder, labels_placeholder
