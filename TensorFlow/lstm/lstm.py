# -*- coding: utf-8 -*-
# lstm 语言模型
#
# Author: Igor

import time

import numpy as np
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
flags.DEFINE_float('max_epoch', 10,
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


class LanguageLSTM():
    '''
    LSTM 语言模型
    '''

    def __init__(self, config, session, is_training):
        '''
        初始化模型
        :param config: 模型参数配置
        :param session: tensorflow session
        :param reader : 数据的读取器
        '''
        self._config = config
        self._session = session
        self._is_traing = is_training

        # self._initial_state = None
        # self._lr = None
        # self._train_op = None
        # self._final_state = None
        # self._cost = None
        # self.summary_writer = None

    def inference(self):
        '''
        inference
        '''
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self._config.hidden_size, forget_bias=0.0)
        if self._is_traing and self._config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell,
                output_keep_prob=self._config.keep_prob
            )
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * self._config.num_layers)

        self._initial_state = cell.zero_state(
            self._config.batch_size,
            tf.float32
        )

        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable(
                'embedding',
                [self._config.vocab_size, self._config.hidden_size]
            )
            inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)

        if self._is_traing and self._config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs,
                                   self._config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("LSTM"):
            for time_step in range(self._config.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(
            tf.concat(1, outputs), [-1, self._config.hidden_size])
        softmax_w = tf.get_variable("softmax_w",
                                    [self._config.hidden_size,
                                     self._config.vocab_size])
        softmax_b = tf.get_variable("softmax_b",
                                    [self._config.vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b

        self._final_state = state

        return logits

    def loss(self, logits):
        '''
        计算损失
        :param logits:
        '''
        loss = tf.nn.seq2seq.sequence_loss_by_example(  # 损失
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones(
                [self._config.batch_size * self._config.num_steps])])
        return loss

    def optimize(self, loss):
        self._cost = cost = tf.reduce_sum(loss) / self._config.batch_size
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          self._config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)  # 梯度下降
        self._train_op = train_op = optimizer.apply_gradients(zip(grads, tvars))  # 更新

        return train_op

    def assign_lr(self, lr_value):
        self._session.run(tf.assign(self.lr, lr_value))

    def run_epoch(self, data, reader, summary_op, verbose=False):
        '''
        Runs the model on given data
        :param data: 数据
        :param eval_op: 计算操作
        :param verbose:
        :return: costs
        '''
        epoch_size = (len(data) // self._config.batch_size - 1) // self._config.num_steps
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = self._initial_state.eval()
        for step, (x, y) in enumerate(reader.iterator(data, self._config.batch_size, self._config.num_steps)):
            feed_dict = {self.input_data: x, self.targets: y, self.initial_state: state}
            cost, state, _ = self._session.run([self.cost, self.final_state, self.train_op],
                                               feed_dict)
            costs += cost
            iters += self._config.num_steps

            if verbose and step % (epoch_size // 10) == 10:
                print("%.3f ; perplexity: %.3f ; speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters),
                       iters * self._config.batch_size / (time.time() - start_time)))

                print("Summary Wrtier")
                summary_str = self._session.run(summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)

        return np.exp(costs / iters)

    def train(self, data, reader):
        if not FLAGS.data_path:
            raise ValueError("Must set --data_path to data directory")
        # self.build_graph()

        with tf.Graph().as_default(), tf.Session() as session:
            self._session = session
            # 初始化所有变量
            initializer = tf.random_normal_initializer(
                -self._config.init_scale, self._config.init_scale)

            self._input_data = tf.placeholder(
                tf.int32, [config.batch_size, config.num_steps])
            self._targets = tf.placeholder(
                tf.int32, [config.batch_size, config.num_steps])

            # 推理
            logits = self.inference()

            # 计算损失
            loss = self.loss(logits)

            # 最优化
            self.optimize(loss)

            summary_op = tf.merge_all_summaries()

            # saver = tf.train.Saver()

            self.summary_writer = tf.train.SummaryWriter(FLAGS.data_path,
                                                         graph_def=self._session.graph_def)
            tf.initialize_all_variables().run()
            for i in range(self._config.max_max_epoch):
                lr_decay = self._config.lr_decay ** max(i - self._config.max_epoch, 0.0)
                self.assign_lr(self._config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1,
                                                         self._session.run(self.lr)))
                train_perplexity = self.run_epoch(data, reader, summary_op, verbose=True)

    def load(self):
        '''
        载入模型
        '''
        pass

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


if __name__ == '__main__':
    # from TensorFlow.word_rnn import reader
    #
    # train = test.ptb_raw_data('data/')[0]
    # config = Options()
    # session = tf.Session()
    #
    # lstm = LanguageLSTM(config, session, True)
    # lstm.train(data=train, reader=test)
    pass
