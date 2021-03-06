# -*- coding: utf-8 -*-
#  
#
# Author: Igor

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf

from tensorgo.cnnc import cnnc
from tensorgo.cnnc import TRAIN_PATH

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', TRAIN_PATH,
                           'directory where to write envent log and'
                           'checkpoint')
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            'Number of batches to run')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Where to log device placement.')


def train():
    '''
    训练 cnnc 模型
    '''
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # 传入数据所在的文件夹
        features, labels = cnnc.distorted_inputs(FLAGS.train_dir)


        logits = cnnc.inference(features)

        loss = cnnc.loss(logits, labels)

        train_op = cnnc.train(loss, global_step)

        # saver
        saver = tf.train.Saver(tf.all_variables())

        # 构建总结操作
        summary_op = tf.merge_all_summaries()

        # 初始化操作
        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            # 模型收敛
            assert not np.isnan(loss_value), "Model diverged with loss = NAN"

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
