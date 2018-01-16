#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 上午11:18
# @Author  : yizhen
# @Site    : 
# @File    : sort_pos.py
# @Software: PyCharm

import codecs
import sys
if __name__ == '__main__':
    dataset = codecs.open(sys.argv[1], 'r', encoding='utf-8').read().strip().split('\n')  #['','',]
    #print (dataset)

    for data in sorted(dataset, key=lambda d: ''.join([str.split('_')[0] for str in d.split()])):  #

        print(data, end='\n')
        #sorted(lst, key=lambda i: -len())
    #d = dataset[0]



