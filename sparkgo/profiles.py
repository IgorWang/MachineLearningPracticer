#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-6-14
'''
Computing User Profiles with Spark
data:
    /data/tracks.csv
    /data/cust.csv
'''
import os
import csv

# os.environ["PYSPARK_PYTHON"] = "/home/igor/anaconda3/bin/ipython"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.mllib.stat import Statistics
except ImportError as e:
    print("Error importing Spark Modules", e)

TRACKS_PATH = '/data/tracks.csv'
CUST_PATH = '/data/cust.csv'


def make_tracks_kv(str):
    l = str.split(",")
    return [l[1], [[int(l[2]), l[3], int(l[4]), l[5]]]]


def compute_stats_byuser(tracks):
    mcount = morn = aft = eve = night = 0
    tracklist = []
    for t in tracks:
        trackid, dtime, mobile, zip_code = t
        if trackid not in tracklist:
            tracklist.append(trackid)
        d, t = dtime.split(" ")
        hourofday = int(t.split(":")[0])
        mcount += mobile
        if (hourofday < 5):
            night += 1
        elif (hourofday < 12):
            morn += 1
        elif (hourofday < 17):
            aft += 1
        elif (hourofday < 22):
            eve += 1
        else:
            night += 1
        return [len(tracklist), morn, aft, eve, night, mcount]


def main():
    conf = SparkConf()
    conf.setMaster("spark://192.168.199.123:8070")
    conf.setAppName("User Profile Spark")

    sc = SparkContext(conf=conf)
    print("connection sucessed with Master", conf)
    data = [1, 2, 3, 4]
    distData = sc.parallelize(data)
    print(distData.collect())
    #
    raw = open(TRACKS_PATH, 'r').read().split("\n")
    tackfile = sc.parallelize(raw)

    tackfile = tackfile.filter(lambda line: len(line.split(',')) == 6)
    tbycust = tackfile.map(lambda line: make_tracks_kv(line)).reduceByKey(lambda a, b: a + b)

    custdata = tbycust.mapValues(lambda a: compute_stats_byuser(a))

    print(custdata.first())
    # # compute aggreage stats for entire track history
    # # aggdata = Statistics.colStats(custdata.map(lambda x: x[1]))
    # #
    # # print(aggdata)
    # print("SUCCESS")


if __name__ == '__main__':
    main()
