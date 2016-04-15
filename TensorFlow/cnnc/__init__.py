# -*- coding: utf-8 -*-
#  
#
# Author: Igor
import os

# 数据所在的路径
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
NEG_PATH = os.path.join(DATA_PATH, 'review_polarity', 'rt-polarity.neg')
POS_PATH = os.path.join(DATA_PATH, 'review_polarity', 'rt-polarity.pos')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
