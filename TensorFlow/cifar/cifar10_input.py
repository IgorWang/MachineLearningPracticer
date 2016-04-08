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

    如果需要N个通道并行的读取,调用这个函数N次.将会生成N个独立的读取器,
    独立的读取文件中不同的文件和位置,将会生成更好的混合样本

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

    # 从record_bytes中取出标签
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 取出图像
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.depth])
    # 从[depth,height,width] 转置为 [height,width,depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    '''
    构造批量图像和标签的队列
    :param image:3-D Tensor of [height,width,3]
    :param label:1-D Tensor
    :param min_queue_examples:队列中保持的最少样本数量
    :param batch_size:一批图像的数量

    :return:
        images:Images 4D tensor [batch_size,height,width,3]
        labels:Labels 1D tensor [batch_size]
    '''

    # min_after_dequeue 定义了随机抽样的缓冲buffer大小
    # capacity必须比min_after_dequeue更大,取决于预取出样本数量的大小
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        num_threads=num_preprocess_threads)

    # Display the training images in the visualizer
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


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
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 图像的预处理,对图像进行随机的变形,提高图像识别的精确度和范化能力
    # 裁剪图像的像素
    distored_image = tf.random_crop(reshaped_image, [height, width, 3])
    # 翻转图像
    distored_image = tf.image.random_flip_left_right(distored_image)

    distored_image = tf.image.random_brightness(distored_image, max_delta=63)
    distored_image = tf.image.random_contrast(distored_image, lower=0.2, upper=1.8)

    # 像素的正规化,减去均值,除以方差
    float_image = tf.image.per_image_whitening(distored_image)

    # 保证随机的shuffing有好的混合性质
    min_fraction_of_example_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_example_in_queue)

    print('Filling queue with %d CIFAR images before starting to train.'
          'This will take a few minutes.' % min_queue_examples)

    # 通过构建样本队列批量生成图像和标签
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
    '''
    利用Reader ops CIFAR评测的输入

    :param eval_data:bool,指明训练数据或测试数据用于预测
    :param data_dir:Path to the CIFAR-10 data directory
    :param batch_size:批量大小
    :return:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    '''
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 构建文件队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 读取样本
    read_input = read_cifar10(filename_queue)
    reshape_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 裁剪图像
    resized_image = tf.image.resize_image_with_crop_or_pad(reshape_image,
                                                           width, height)

    # normalization
    float_image = tf.image.per_image_whitening(resized_image)

    min_fraction_of_example_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_example_in_queue)
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size)
