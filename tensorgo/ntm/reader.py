#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-6-3
import sys
import random

import tensorflow as tf
from tensorflow.models.rnn.translate import data_utils

import numpy as np

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
DATA_DIR = '/data/ntm/'
TRAIN_FN = 'training-giga-fren.tar'
DEV_FN = 'dev-v2.tgz'


def read_data(source_path, target_path, max_size=None):
    '''
    Read data from source and target files
    :param source_path: 原语言token_ids文件
    :param target_path: 目标语言token_ids文件
    :param max_size: 最大读的行数
    :return:a list of length len(_buckets)
    '''
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode='r') as source_file:
        with tf.gfile.GFile(target_path, mode='r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print(" reading data line %d" % counter)
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def get_batch(data, batch_size, bucket_id):
    encoder_size, decoder_size = _buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    for i in range(batch_size):
        encoder_input, decoder_input = random.choice(data[bucket_id])

        # Encoder inputs are padden and the reverses
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "Go" sysmbol,and are padded then
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    for lenth_idx in range(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][lenth_idx]
                      for batch_idx in range(batch_size)], dtype=np.int32)
        )

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in range(batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights


if __name__ == '__main__':
    # data_utils.maybe_download(DATA_DIR, TRAIN_FN, data_utils._WMT_ENFR_TRAIN_URL)
    # data_utils.maybe_download(DATA_DIR, TRAIN_FN, data_utils._WMT_ENFR_DEV_URL)
    source_path, target_path, _, _, _, _ = data_utils.prepare_wmt_data(DATA_DIR, 10000, 10000)
    data_set = read_data(source_path, target_path, 50)
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = get_batch(data_set
                                                                          , 10,0)
