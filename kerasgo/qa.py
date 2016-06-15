#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-6-3

import re
import tarfile
from functools import reduce

import numpy as np
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge
from keras.layers import recurrent
from keras.models import Sequential
from keras.utils.visualize_util import plot

np.random.seed(1337)

DATA_PATH = '/data/task_qa.tar.gz'

tar = tarfile.open(DATA_PATH)

challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'


def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = line.decode()
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = nltk.word_tokenize(q)
            substory = None
            if only_supporting:
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = nltk.word_tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=50):
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word2id, vocab_size, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word2id[w] for w in story]
        xq = [word2id[w] for w in query]
        y = np.zeros(vocab_size)
        y[word2id[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


def main():
    RNN = recurrent.GRU
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 5
    print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE,
                                                               QUERY_HIDDEN_SIZE))
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
    vocab_size = len(vocab) + 1
    word2id = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    X, Xq, Y = vectorize_stories(train, word2id, vocab_size, story_maxlen, query_maxlen)
    tX, tXq, tY = vectorize_stories(test, word2id, vocab_size, story_maxlen, query_maxlen)

    print('vocab = {}'.format(vocab))
    print('X.shape = {}'.format(X.shape))
    print('Xq.shape = {}'.format(Xq.shape))
    print('Y.shape = {}'.format(Y.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

    print('Build model...')

    sentrnn = Sequential()
    sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=story_maxlen))
    sentrnn.add(RNN(SENT_HIDDEN_SIZE, return_sequences=False))

    qrnn = Sequential()
    qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=query_maxlen))
    qrnn.add(RNN(QUERY_HIDDEN_SIZE, return_sequences=False))

    model = Sequential()
    model.add(Merge([sentrnn, qrnn], mode='concat'))
    model.add((Dense(vocab_size, activation='softmax')))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # plot(model)
    print("Training")
    model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS,
              validation_split=0.05, verbose=True)
    loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


if __name__ == '__main__':
    # data = get_stories(tar.extractfile(challenge.format('train')))
    # print(data[0])
    main()
