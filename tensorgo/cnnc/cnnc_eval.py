# -*- coding: utf-8 -*-
#  
#
# Author: Igor
from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf

from tensorgo.cnnc import cnnc
from tensorgo.cnnc import TRAIN_PATH, DATA_PATH

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', os.path.join(DATA_PATH, 'eval'),
                           'eval eirectory')
tf.app.flags.DEFINE_string("checkpoint_dir", TRAIN_PATH,
                           "Directory where to read model checkpoints")
tf.app.flags.DEFINE_integer("eval_interval_secs", 5,
                            "How often to run the eval.")
tf.app.flags.DEFINE_integer("num_examples", 1000,
                            "Number of examples to run")
tf.app.flags.DEFINE_boolean("run_once", False,
                            "where to run eval only once")


def evaluate(input_x, input_y):
    '''
    评价 文本分类
    :return
        result:预测的结果,哪一维更大
        accuracy:精确度
    '''
    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        # 恢复模型
        features = tf.placeholder(tf.int32, [None, cnnc.SEQUENCE_LENGTH])
        labels = tf.placeholder(tf.int32, [None, cnnc.FLAGS.num_class])
        logits = cnnc.inference(features)
        predictions = tf.arg_max(logits, 1)
        correct_predictions = tf.equal(predictions, tf.arg_max(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                          dtype=tf.float32))
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("SUCESS")
        else:
            print("No checkpoint file found")

        result, accuracy = sess.run([predictions, accuracy], feed_dict={features: input_x, labels: input_y})

    return result, accuracy


if __name__ == '__main__':
    import pandas as pd

    filenames = os.path.join(TRAIN_PATH, 'train.txt')

    data = pd.read_csv(filenames, sep=',', header=None)
    x = data.iloc[0:100, 0:cnnc.SEQUENCE_LENGTH].values
    y = data.iloc[0:100, cnnc.SEQUENCE_LENGTH:].values

    result = evaluate(x, y)
    print(result[1])
