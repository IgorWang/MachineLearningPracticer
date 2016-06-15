#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-5-31
'''
开始的样例
'''
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
import numpy as np

# Sequential表示了一个顺序的神经网络模型
# 数据顺序的传播
model = Sequential()

# 构建模型的过程
model.add(Dense(output_dim=200, input_dim=784))
model.add(Activation("relu"))
model.add(Dense(output_dim=200))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, newshape=(60000, -1))
X_test = np.reshape(X_test, newshape=(X_test.shape[0], -1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model.fit(X_train, y_train, nb_epoch=100, batch_size=32,
          validation_data=(X_test,y_test))
