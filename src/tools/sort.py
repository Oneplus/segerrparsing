#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/19 下午9:16
# @Author  : yizhen
# @Site    : 
# @File    : sort.py
# @Software: PyCharm
import codecs
import sys


if __name__ == "__main__":
    dataset = codecs.open(sys.argv[1], 'r', encoding='utf-8').read().strip().split('\n\n')  #['','',]
    #print (dataset)
    for data in sorted(dataset, key=lambda d: ''.join([line.split()[1] for line in d.split('\n')])):  #

        print(data, end='\n\n')
        #sorted(lst, key=lambda i: -len())
    #d = dataset[0]
    #print ([''.join([line.split()[1] for line in d.split('\n')])])