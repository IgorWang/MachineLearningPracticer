#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-6-14
import os
import sys

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
except ImportError as e:
    print("Error importing Spark Modules", e)

conf = SparkConf()
# conf.setMaster("spark://192.168.199.123:8070")
conf.setAppName("Test Spark")

sc = SparkContext(conf=conf)
print("connection sucessed with Master", conf)
data = [1, 2, 3, 4]
distData = sc.parallelize(data)
print(distData.collect())
