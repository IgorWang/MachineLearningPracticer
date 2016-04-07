# -*- coding: utf-8 -*-
#  
#
# Author: Igor

import time
import os

import numpy as np
import tensorflow as tf

from TensorFlow.word_rnn.configure import *


class WordRNN(object):
    '''
    基于字的LSTM模型
    '''

    def __init__(self, config, is_training):

        self.config = config
        self.is_training = is_training

        # 模型的lstm cell样式
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.config.hidden_size, forget_bias=0.0)

        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell,
                output_keep_prob=self.config.keep_prob)

        # 内部的RNN模型
        self.cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * config.num_layers)

        self.initial_state = self.cell.zero_state(
            config.batch_size, tf.float32)

        # 模型输入占位符
        self.input_data = tf.placeholder(tf.int32,
                                         [config.batch_size, config.num_steps],
                                         name="input_words")
        self.targets = tf.placeholder(tf.int32,
                                      [config.batch_size, config.num_steps],
                                      name="target_words")

        self.logits, self.final_state = self.inference()

        if not is_training:
            return

        loss = self.loss(self.logits)
        self.optimize(loss)

    def inference(self):
        '''

        '''
        config = self.config

        with tf.variable_scope("embedding"):
            with tf.device('/cpu:0'):
                self.embedding = tf.get_variable(
                    "embedding",
                    [config.vocab_size, config.hidden_size])
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

            if self.is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("LSTM"):
            for time_step in range(config.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(1, outputs),
                            shape=[-1, config.hidden_size])

        with tf.variable_scope("output"):
            softmax_w = tf.get_variable("softmax_w",
                                        [config.hidden_size,
                                         config.vocab_size])

            softmax_b = tf.get_variable("softmax_b",
                                        [config.vocab_size])
            logits = tf.matmul(output, softmax_w) + softmax_b

        return logits, state

    def loss(self, logits):
        loss = tf.nn.seq2seq.sequence_loss_by_example(  # 损失
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones(
                [self.config.batch_size * self.config.num_steps])])
        return loss

    def optimize(self, loss):
        self.cost = cost = tf.reduce_sum(loss) / self.config.batch_size

        tf.scalar_summary("cost", cost)

        self.lr = tf.Variable(0.0, trainable=False)

        tf.scalar_summary("learning_rate", self.lr)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          self.config.max_grad_norm)

        # tf.histogram_summary("gradients", grads)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)  # 梯度下降
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))  # 更新

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def run_epoch(self, session, reader, data, verbose=False):
        """
        run one epoch based on data
        """
        epoch_size = ((len(data) // self.config.batch_size) - 1) // self.config.num_steps
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = self.initial_state.eval()  # 初始状态
        for step, (x, y) in enumerate(reader.iterator(
                data, self.config.batch_size, self.config.num_steps)):
            feed_dict = {self.input_data: x, self.targets: y, self.initial_state: state}
            cost, state, _ = session.run([self.cost, self.final_state, self.train_op],
                                         feed_dict)
            costs += cost
            iters += self.config.num_steps

            if verbose and step % (epoch_size // 10) == 10:
                print("%.3f ; perplexity : %.3f speed : %.2f wps " % (
                    step * 1.0 / epoch_size,
                    np.exp(costs / iters),
                    time.time() - start_time))

                summary_str = session.run(self.summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)

        return np.exp(costs / iters)

    @classmethod
    def train(cls, config, reader, verbose=False):
        '''
        :reader 数据读取器
        '''

        if not FLAGS.data_path:
            raise ValueError("Must set --data_path to WordRNN model")

        raw_data, _ = reader.read_data(FLAGS.data_path)
        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

            with tf.variable_scope("model", reuse=None, initializer=initializer):
                # 初始化模型
                model = cls(config, is_training=True)

            cls.summary_op = tf.merge_all_summaries()

            cls.saver = tf.train.Saver(tf.trainable_variables())

            cls.summary_writer = tf.train.SummaryWriter(FLAGS.data_path,
                                                        graph_def=session.graph_def)

            # 初始化所有变量
            tf.initialize_all_variables().run()

            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                model.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d ; Learning rate %.3f " %
                      (i + 1, session.run(model.lr)))

                epoch_perplexity = model.run_epoch(session, reader, raw_data, verbose=True)

                if verbose and (i % (config.max_max_epoch // 10) == 0 or i == config.max_max_epoch - 1):
                    print("%.3f finish ; perplexity %.3f"
                          % (i / config.max_max_epoch, epoch_perplexity))
                    chechkpoint_path = os.path.join(FLAGS.data_path, "model.ckpt")
                    model.saver.save(sess=session, save_path=chechkpoint_path, global_step=i)
                    print("save model to {}".format(chechkpoint_path))
        return model

    @classmethod
    def load(cls, config, model_path):
        '''
        载入模型
        '''
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = cls(config, is_training=False)
        model.sess = tf.Session()

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Read model from %s" % ckpt.model_checkpoint_path)
            saver.restore(model.sess, ckpt.model_checkpoint_path)
            print("Model loaded")
        else:
            print("No checkpoint file found")
        return model

    def predict(self, x):
        # session = tf.Session()
        # tf.initialize_all_variables().run(session=session)

        state = self.initial_state.eval(session=self.sess)
        feed_dict = {self.input_data: x, self.initial_state: state}
        logits, final_state = self.sess.run([self.logits, self.final_state], feed_dict)

        probs = tf.nn.softmax(logits)
        predict_result = tf.arg_max(probs, dimension=1)

        predict_result = self.sess.run(predict_result)

        return predict_result


if __name__ == '__main__':
    from TensorFlow.word_rnn import reader

    FLAGS.data_path = os.path.join(os.path.dirname(__file__), 'data')
    config = Options()
    data, word2id = reader.read_data(FLAGS.data_path)
    config.vocab_size = len(word2id)
    print(len(word2id))
    # model = WordRNN.train(config, reader, verbose=True)

    # _______________________________________

    #predict
    config.batch_size = 1
    model = WordRNN.load(config, 'data/')

    data, word2id = reader.read_data(FLAGS.data_path)
    id2word = {v: k for k, v in word2id.items()}
    epoch = 0

    for i, (x, y) in enumerate(reader.iterator(data, config.batch_size, config.num_steps)):
        if i > 30:
            print("x      :", list(map(lambda x: id2word[x], x[0])))
            pred = model.predict(x)
            print("predict:", list(map(lambda x: id2word[x], pred)))
            if i > 60:
                break


