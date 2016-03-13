__author__ = 'igor'

"""Trains and Evaluates the MNIST netword using a feed dictionary"""

import os.path
import time

import tensorflow.python.platform
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from TensorFlow.mnist import mnist

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
    labels_placeholder = tf.placeholder(tf.int32,
                                        shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    '''
    Fills the feed_dict for training the given step
    :param data_set: The set of images and labels
    :param images_pl: placeholder
    :param labels_pl: placeholder
    :return:
        feed_dict:the fedd diction mapping from placeholders to values
    '''
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }
    return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('Num examples: %d Num correct: %d Precison @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    """
    Train MNIST for a number of steps
    :return:
    """

    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

    with tf.Graph().as_default():
        # 生成placeholder
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # 构建图模型 计算预测值
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        # 在图中增加损失计算
        loss = mnist.loss(logits, labels_placeholder)

        # 在图中增加和计算梯度
        train_op = mnist.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation
        eval_correct = mnist.evalution(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints
        saver = tf.train.Saver()

        # 构建一个session执行Ops
        sess = tf.Session()

        init = tf.initialize_all_variables()
        sess.run(init)

        # 实例化SummaryWriter,输出summaries和Graph
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        # after everything is built, start the training loop
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)

            # 执行一步训练 返回值是train_op 和 loss
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the sumaries and print an overivew fairly often
            if step % 100 == 0:
                print('Step %d : loss = %.2f ( %3.f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                print('Training Data Eval:')
                do_eval(sess, eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                print('Test Data Eval:')
                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test
                )


def main():
    run_training()


if __name__ == '__main__':
    main()
