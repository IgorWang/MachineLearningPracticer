# -*- coding: utf-8 -*-
#  
#
# Author: Igor
import os

import tensorflow as tf

SEQUENCE_LENGTH = 58
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000


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
    record_defaults = [[1] for i in range((SEQUENCE_LENGTH + 2))]
    record = tf.decode_csv(value,
                           record_defaults=record_defaults)
    # 特征
    result.feature = tf.pack(record[0:SEQUENCE_LENGTH])
    # 标签
    result.label = tf.pack(record[SEQUENCE_LENGTH:])

    return result


def _generate_features_and_labels_batch(feature, label, min_queue_examples, batch_size, shuffle):
    '''
    构建特征和标签的批量队列
    :param features: 1-D Tensor of [SEQUENCE_LENGTH]
    :param lables: 2-D Tensor of [0,1] or [1,0]
    :param min_queue_examples:
    :param batch_size:批量的大小
    :param shuffle:
    :return:
        features:2-D [batch_size,SEQUENCE_LENGTH]
        labels:2-D [batch_size,2]
    '''
    num_preprocess_threads = 1
    if shuffle:
        features, label_batch = tf.train.shuffle_batch(
            [feature, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        features, label_batch = tf.train.batch(
            [feature, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return features, label_batch


def distorted_inputs(filenames, batch_size):
    '''
    构建文本输入 - 训练
    :param data_dir: 数据所在目录
    :param batch_size: 批量大小
    :return:
        document: 2D tensor of [batch_size,SEQUENCE_LENGTH]
        labels: 2D tensor [batch_size,2]
    '''
    # for f in filenames:
    #     if not os.path.exists(f):
    #         raise ValueError("Failed to find file:" + f)

    # 文件的读取队列
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_data(filename_queue)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d case before starting to train.'
          'This will take a few minutes.' % min_queue_examples)

    return _generate_features_and_labels_batch(read_input.feature, read_input.label,
                                               min_queue_examples, batch_size,
                                               shuffle=True)


if __name__ == '__main__':
    filenames = ['data/train/train.txt']
    input_data = distorted_inputs(filenames, 128)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(10000):
            features, lables = sess.run([input_data[0],
                                         input_data[1]])
            print(features)
            print(lables)
        print(features.shape)
        print(lables.shape)

        coord.request_stop()
        coord.join(threads)
