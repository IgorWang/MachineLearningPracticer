__author__ = 'igor'
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    '''
    sigmoid函数的倒函数
    :param z:
    :return:
    '''
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)  # 神经网络的层数
        self.sizes = sizes  # sizes的每个元素表示每一层的大小
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # 初始偏置
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # 初始权值

    def feedforword(self, a):
        '''
        返回神经网络的输出值，如果a是输入
        :param a:
        :return:
        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        利用随机梯度下降方法训练神经网络。
        :param training_data:a list of tuples (x,y)
        :param epochs: 迭代次数
        :param mini_batch_size: 取样大小
        :param eta: 学习速率
        :param test_data: 训练集
        :return:
        '''

        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''
        更新神经网络的权值和偏置
        随机梯度下降
        向后传播
        :param mini_batch: a list of tuples(x,y)
        :param eta: learning rate
        :return:
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # 求偏导
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforword(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)

    def backprop(self, x, y):
        '''
        返回一个(nabla_b,nabla_w)元组,表示C_x函数的梯度
        :param x:
        :param y:
        :return:

        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 向前chuanbo
        actiation = x
        actiations = [x]  # 存贮所有激活的列表

        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, actiation) + b
            zs.append(z)
            actiation = sigmoid(z)
            actiations.append(actiation)

        # 向后传播
        delta = self.cost_derivative(actiations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, actiations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, actiations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        '''
        返回偏导向量 \partial C_x / \partial a for the output activations.
        :param output_activations:
        :param y:
        :return:
        '''
        return (output_activations - y)
