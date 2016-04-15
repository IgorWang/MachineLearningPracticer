# -*- coding: utf-8 -*-
#  
#
# Author: Igor
import numpy as np
import pickle
from collections import Counter
import itertools
import re

from TensorFlow.cnnc import *


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def load_data_and_labels():
    '''
    载入数据
    :return:
    '''

    # Load
    pos_examples = list(open(POS_PATH, 'rb').readlines())
    pos_examples = [str(s).strip() for s in pos_examples]
    neg_examples = list(open(NEG_PATH, 'rb').readlines())
    neg_examples = [str(s) for s in neg_examples]

    # 切分
    train_text = pos_examples + neg_examples
    train_text = [clean_str(sent).split(' ') for sent in train_text]

    # 生成标签 [0,1] 表示正
    pos_labels = [[0, 1] for _ in pos_examples]
    neg_labels = [[1, 0] for _ in neg_examples]
    y = np.concatenate([pos_labels, neg_labels], axis=0)
    return [train_text, y]


def pad_sentences(sentences, padding_word='<PAD/>'):
    '''
    填补句子
    :param sentences:训练句子
        ：:type list[list]
    :param padding_word:填补的标识
    :return: 填补后的句子
    '''
    sequence_length = max(len(x) for x in sentences)
    paded_sentences = []
    for sent in sentences:
        need_pad = sequence_length - len(sent)
        pads = [padding_word] * need_pad
        paded_sentences.append(sent + pads)
    return paded_sentences


def build_vocabulary(sentences):
    '''
    构建词典
    :param sentences:分词后的句子
    :return:词典
    '''
    # 构建词典
    word_counts = Counter(itertools.chain(*sentences))
    # 映射
    # 频次筛选
    vocabulary = [i[0] for i in word_counts.most_common()]
    vocabulary = list(sorted(vocabulary))
    word2id = {x: i for i, x in enumerate(vocabulary)}
    return word2id, vocabulary


def build_input_data(sentences, labels, word2id):
    '''
    映射训练数据
    '''
    x = np.array([[word2id[word] for word in sent] for sent in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    '''
    导入和预处理数据
    :return:
    '''
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    word2id, words_list = build_vocabulary(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, word2id)
    return [x, y, word2id, words_list]


def save_to_csv(data, filename):
    '''
    存储训练数据为csv
    :param data:数据
        :type numpy array
    :return:
    '''
    if not os.path.exists(filename):
        raise FileExistsError("%s don't exists" % filename)
    np.savetxt(filename, data, fmt='%d', delimiter='\t')
    print("save to %s " % filename)


def main():
    x, y, _, _ = load_data()
    csv_file = os.path.join(TRAIN_PATH, 'train.txt')
    data = np.concatenate([x, y], axis=1)
    np.random.shuffle(data)
    save_to_csv(data, csv_file)


if __name__ == '__main__':
    pass
