# -*- coding: utf-8 -*-
#  
#
# Author: Igor

import tensorflow as tf

flags = tf.app.flags

# 定义flags
flags.DEFINE_float('init_scale', 0.1,
                   'the initial scale of the weights')
flags.DEFINE_float('learning_rate', 1.0,
                   'the initial value of the learning rate')
flags.DEFINE_float('max_grad_norm', 2,
                   'the maximum permissible norm of the gradient')
flags.DEFINE_float('num_layers', 2,
                   'the number of LSTM layers')
flags.DEFINE_float('num_steps', 10,
                   'the number of unrolled steps of LSTM')
flags.DEFINE_float('hidden_size', 200,
                   'the number of LSTM units')
flags.DEFINE_float('max_epoch', 15,
                   'the number of epochs trained with the initial learning rate')
flags.DEFINE_float('max_max_epoch', 50,
                   'the total number of epochs for training')
flags.DEFINE_float('keep_prob', 1.0,
                   'the probability of keeping weights in the dropout layer')
flags.DEFINE_float('lr_decay', 0.7,
                   'the decay of the learning rate for each epoch after "max_epoch"')
flags.DEFINE_float('batch_size', 50,
                   'the batch size')
flags.DEFINE_float('vocab_size', 10000, 'the vocab size')
flags.DEFINE_integer('statistics_interval', 5,
                     'Print statistics every n seconds')
flags.DEFINE_integer('summary_interval', 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval.")
flags.DEFINE_integer('checkpoint_interval', 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval.")
flags.DEFINE_string('data_path', 'data/', 'data directory')
flags.DEFINE_string('save_path', 'data/model', 'save directory')

FLAGS = flags.FLAGS


class Options(object):
    """
    Options used by Language Model with LSTM
    """

    def __init__(self):
        '''
        Model options
        :return:
        '''
        self.init_scale = FLAGS.init_scale
        self.learning_rate = FLAGS.learning_rate
        self.max_grad_norm = FLAGS.max_grad_norm
        self.num_layers = FLAGS.num_layers
        self.num_steps = FLAGS.num_steps
        self.hidden_size = FLAGS.hidden_size
        self.max_epoch = FLAGS.max_epoch
        self.max_max_epoch = FLAGS.max_max_epoch
        self.keep_prob = FLAGS.keep_prob
        self.lr_decay = FLAGS.lr_decay
        self.batch_size = FLAGS.batch_size
        self.vocab_size = FLAGS.vocab_size
        self.summary_interval = FLAGS.summary_interval
        self.checkpoint_interval = FLAGS.checkpoint_interval
