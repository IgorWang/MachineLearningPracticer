# -*- coding: utf-8 -*-
#  
#
# Author: Igor

import os
import re

import tensorflow as tf

from TensorFlow.cnnc import cnnc_input
from TensorFlow.cnnc import TRAIN_PATH

SEQUENCE_LENGTH = cnnc_input.SEQUENCE_LENGTH

FLAGS = tf.app.flags.FLAGS

# 模型参数
tf.app.flags.DEFINE_integer('batch_size', 1,
                            'number of text to process in a batch')
tf.app.flags.DEFINE_integer('vocab_size', 20267,
                            'number of words in datasets')
tf.app.flags.DEFINE_integer('embedding_size', 300,
                            'size of word embedding')
tf.app.flags.DEFINE_integer('num_class', 2, 'class numbers')

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cnnc_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

TOWER_NAME = 'tower'

# 卷积参数
# 过滤器的大小
FILTER_SIZES = [2, 3, 4, 5, 6, 7]
# 过滤器的数量
NUM_FILTERS = 20

# 训练过程的全局常量
MOVING_AVERAGE_DECAY = 0.9999  # 移动平均衰减
NUM_EPOCHS_PER_DECAY = 350.0  # 当学习速率开始下降的(期数)Epochs
LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习速率衰减因子
INITIAL_LEARNING_RATE = 0.1  # 初始化学习速率


def distorted_inputs(data_dir):
    filenames = os.listdir(data_dir)
    return cnnc_input.distorted_inputs(filenames, FLAGS.batch_size)


def _variable_on_cpu(name, shape, initializer):
    '''
    辅助函数:在CPU中创建Variable
    :param name:变量名称
    :param shape:形状
    :param initializer:初始化
    :return:
        Variable Tensor
    '''
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    辅助函数:利用权重衰减初始化Variable

    变量利用truncated normal distribution
    只有指定的时候才进行权重衰减(weight decay)
    :param name:变量的名称
    :param shape:形状
    :param stddev:trucated Gaussian的标准差
    :param wd:增加 L2Loss 权重衰减,如果是None,weight decay不被增加

    :return:
        Variabel Tensor
    '''
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        # Stores value in the collection with the given name
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):
    '''
    辅助函数:创建激活神经元的summaries

    提供激活元的直方图的总结
    提供激活元稀疏测量的总结

    :param x:Tensor
    :return:
        nothing
    '''
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    # 0的占比,稀疏性
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inference(features, embedding=None, train_embedding=True):
    '''
    构建 文本分类 模型

    为了变量共享统一用tf.get_variabel()实例化所有variables
    多GPU:tf.get_variable()
    单个GPU:tf.Variable()

    :param features:returned from distored_inputs() or inputs()
    :param embedding:词向量矩阵的初始值
    :param train_embedding:是否继续训练词向量
    :return:
        Logits
    '''

    # Embedding layer
    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
        if embedding:
            initializer = tf.constant_initializer(embedding)
        else:
            initializer = tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_size], -1.0, 1.0)
        embedding = tf.get_variable('embedding',
                                    initializer=initializer, trainable=train_embedding)
        text_embeding = tf.nn.embedding_lookup(embedding, features)
        # 4-D Tensor[batch_size,SEQUENCE_LENGTH,embedding_size,1]
        text_embeding = tf.expand_dims(text_embeding, -1)
        # TensorFlow’s convolutional conv2d operation
        # expects a 4-dimensional tensor with dimensions
        # corresponding to batch, width, height and channel.

    # 为不同过滤器大小创建卷积+maxpool层
    pooled_outputs = []
    for i, filter_size in enumerate(FILTER_SIZES):
        with tf.variable_scope('conv-maxpool-%s' % filter_size) as scope:
            # convolution Layers
            # filter Tensor [filter_height, filter_width, in_channels, out_channels]
            filter_shape = [filter_size, FLAGS.embedding_size, 1, NUM_FILTERS]
            kernel = _variable_with_weight_decay('weights',
                                                 shape=filter_shape,
                                                 stddev=1e-4, wd=0.0)
            biases = _variable_on_cpu('biases', shape=[NUM_FILTERS],
                                      initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(text_embeding, kernel, [1, 1, 1, 1],
                                padding='VALID')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv1)

            pool = tf.nn.max_pool(conv1,
                                  ksize=[1, SEQUENCE_LENGTH - filter_size + 1, 1, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID',
                                  name=scope.name)
            pooled_outputs.append(pool)
    # 结合所有的pooled features
    num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
    h_pool = tf.concat(3, pooled_outputs)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    with tf.variable_scope("output") as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_filters_total, FLAGS.num_class],
                                              stddev=1 / num_filters_total, wd=0.0)
        bias = _variable_on_cpu('biases', [FLAGS.num_class],
                                initializer=tf.constant_initializer(0.0))

        logits = tf.add(tf.matmul(h_pool_flat, weights),
                        bias, name=scope.name)
        _activation_summary(logits)

    return logits


def loss(logits, labels):
    '''
    对所有训练的变量添加L2Loss
    计算单批次的平均交叉熵损失

    为 "Loss" 和 "Loss/avg"添加总结
    :param logits:Logits from inference()
    :param labels:Labels from distored_inputs() or inputs() 1-D tensor
    :return:
        Loss tensor of type float
    '''
    labels = tf.cast(labels, tf.float32)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels,
                                                            name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # 总的损失定义为交叉熵损失加熵所有权重衰减的损失
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    '''
    为CIFAR模式添加损失总结
    生成所有损失和相关总结的移动平均:为可视化网络的性能
    :param total_loss:Total loss from loss()
    :return:
        loss_averages_op:op for generating moving averages of losses.
    '''

    # 为单个的损失和总共的损失计算指数移动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + '(raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    '''
    训练 文本分类 模型
    创建一个optimizer应用于所有训练的变量
    对所有的训练变量移动平均
    :param total_loss:单批次总的损失Loss()
    :param global_step:Integer 变量,计算训练步骤过程的数量
    :return:
        train_op : 训练的操作 op
    '''
    # 影响学习速率的变量,每一期的批次数量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # 基于步骤的学习速率指数衰减
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    tf.scalar_summary('learning_rate', lr)

    # 生成所有损失和相关总结的移动平均
    loss_averages_op = _add_loss_summaries(total_loss)

    # 计算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 添加训练变量的直方图
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # 梯度直方图
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # 追踪训练变量的指数移动平均
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
