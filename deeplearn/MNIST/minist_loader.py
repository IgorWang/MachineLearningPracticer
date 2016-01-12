__author__ = 'igor'
""""
minist__loader
'"""

import pickle
import gzip

import numpy as np


def load_data():
    '''
    返回 MINIST的数据,
    :return:一个包含训练数据，验证数据，测试数据的元组
    '''
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


def vectorized_result(y):
    '''
    对y进行向量化，转化为10-demensional unit
    :param y:
    :return:
    '''
    e = np.zeros((10, 1))
    e[y] = 1
    return e


def load_data_wrapper():
    '''
    基于load_data,格式更容易被使用
    :return:
    '''
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


if __name__ == '__main__':
    tr_d, va_d, te_d = load_data()
    print(te_d[0].shape)

    # tr_d,va_d,te_d = load_data_wrapper()
    # print(tr_d);
    print(__path__)
