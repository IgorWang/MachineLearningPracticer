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

from tensorgo.cifar import cifar10_input

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
MOVING_AVERAGE_DECAY = 0.9999  # 移动平均衰减
NUM_EPOCHS_PER_DECAY = 350.0  # 当学习速率开始下降的(期数)Epochs
LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习速率衰减因子
INITIAL_LEARNING_RATE = 0.1  # 初始化学习速率

TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def distorted_inputs():
    '''
    利用Reader ops为训练构建CIFAR数据集的输入
    :return:
        images:Images 4D tensor of [batch_size,IMAGE_SIZE,IMAGE_SIZE,3]
        labels:Labels 1D [batch_size]
    '''
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.distorted_inputs(data_dir, FLAGS.batch_size)


def inputs(eval_data):
    '''
    利用Reader ops为评价CIFAR构建输入
    :param eval_data:是否利用测试集
    :return:
        images:Images 4D tensor of [batch_size,IMAGE_SIZE,IMAGE_SIZE,3]
        labels:Labels 1D [batch_size]
    '''
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.inputs(eval_data, data_dir, FLAGS.batch_size)


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


def inference(images):
    '''
    构建 CIFAR-10 模型

    为了变量共享统一用tf.get_variabel()实例化所有variables
    多GPU:tf.get_variable()
    单个GPU:tf.Variable()

    :param images: Images returned from distored_inputs() or inputs()
    :return:
        Logits.
    '''

    # conv1 卷积层1
    with tf.variable_scope('conv1') as scope:
        # filter Tensor [filter_height, filter_width, in_channels, out_channels]
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                             stddev=1e-4, wd=0.0)

        # 输出:和输出相同类型的Tensor
        # [batch, out_height, out_width, filter_height * filter_width * in_channels]
        # For each patch, right-multiplies the filter matrix and the image patch vector.
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.0001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192],
                                  initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,
                            name=scope.name)
        _activation_summary(local4)

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASS],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASS],
                                  initializer=tf.constant_initializer(0.0))
        # softmax(WX+b)
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


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
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                   labels,
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
    训练 CIFAR-10 模型
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
    maybe_download_and_extract()
