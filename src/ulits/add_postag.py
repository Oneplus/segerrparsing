#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 下午2:21
# @Author  : yizhen
# @Site    : 
# @File    : add_postag.py.py
# @Software: PyCharm

import codecs
import argparse

def main():
    cmd = argparse.ArgumentParser("add postag")
    cmd.add_argument("--origin_path", type = str, help = "origin_path be converted",default='../../data/postag_test/input.txt')
    cmd.add_argument("--target_path", type = str, help = "converted target", default='../../data/postag_test/output.txt')

    args = cmd.parse_args()
    res = []
    with codecs.open(args.origin_path, 'r', encoding='utf-8') as fp:
        with codecs.open(args.target_path, 'w', encoding='utf-8') as fp_targt:
            fp_origin = fp.read().strip().split('\n')
            for sentence in fp_origin:
                temp = [word+"_TEST"for word in sentence.split()]
                res.append(' '.join(temp))
            fp_targt.write('\n'.join(res))

if __name__ == '__main__':
    main()