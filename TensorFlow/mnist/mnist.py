__author__ = 'igor'

"""
构建 mnist network
构建 Graph

1.inference() - Builds the model as far as is required for running the network
forward to make predictions.
2.loss() -Adds to the inference model the layers required to generate loss
3.training() - Adds to the loss model the Ops required to generate and
apply gradients.
"""

import os.path
import math

import tensorflow.python.platform
import tensorflow as tf

# THE MNIST dataset has 10 classes
NUM_CLASSES = 10

# MNIST 的图像是28×28 pixedls
IMAGE_SIZE = 28
# 特征的维度
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units, hidden2_units):
    '''
    构建 MNIST model,向前传播
    :param images: Image placeholder,输入
    :param hidden1_units: 第一个隐藏层的大小
    :param hidden2_units: 第二个隐藏层的大小
    :return:
        softmax_linear:Output tensor with the computed logits.
    '''

    # Hidden 1
    with tf.name_scope("hidden1"):
        weights = tf.Variable(  # 输入层到输出层的weights
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name="weights")
        biases = tf.Variable(
            tf.zeros([hidden1_units]),
            name='biases'
        )
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)  # 激活函数是rectifier
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('soft_max_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases  # 激活层是横等函数

    return logits


def loss(logits, labels):
    '''
    从logits 和labels 计算损失
    :param logits: Logits tensor,float-[batch_size,NUM_CLASS]
    :param labels: Labels tensor,int32-[batch_size]
    :return:Loss tensor
    '''
    # 用one-hot的方式对labels_placeholder进行编码
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    one_hot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            one_hot_labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    '''
    设置 training Ops
    :param loss:
    :param learning_rate:
    :return:
    '''
    tf.scalar_summary(loss.op.name, loss)
    # 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evalution(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


if __name__ == '__main__':
    pass
