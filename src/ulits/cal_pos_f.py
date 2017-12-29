#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/24 下午1:58
# @Author  : yizhen
# @Site    : 
# @File    : cal_pos_f.py
# @Software: PyCharm

import codecs
import argparse

def convert(sentence):
    '''

    convert sentence to mid representation (word, start-id_end-id, pos)
    :param sentence: type: word_pos
    :return:
    '''
    mid_representation = []
    sentence = sentence.strip().split()
    start_id, end_id = 0, 0
    for word_pos in sentence:
        word, pos = word_pos.split('_')
        end_id += len(word)
        start_end_id = str(start_id) + '_' + str(end_id)
        temp = (word, start_end_id, pos)
        mid_representation.append(temp)
        start_id = end_id

    return mid_representation


def main():
    cmd = argparse.ArgumentParser('for calcutating pos f')
    cmd.add_argument('--gold_pos_path', help = "gold_pos_path for calcutating pos f", type = str, default='../../data/test_pos_f/gold.txt')
    cmd.add_argument('--auto_pos_path', help = "auto_pos_path for calcutation pos f", type = str, default='../../data/test_pos_f/auto.txt')

    args = cmd.parse_args()

    with codecs.open(args.gold_pos_path, 'r', encoding='utf-8') as fp_gold:
        gold_pos = fp_gold.read().strip().split('\n')

    with codecs.open(args.auto_pos_path, 'r', encoding='utf-8') as fp_auto:
        auto_pos = fp_auto.read().strip().split('\n')

    assert len(auto_pos) == len(gold_pos)
    n_corr, n_pred, n_gold = 0.0, 0.0, 0.0
    for index_gold, gold in enumerate(gold_pos):
        mid_representation_gold = convert(gold)
        # print(mid_representation_gold)
        mid_representation_auto = convert(auto_pos[index_gold])
        for auto in mid_representation_auto:
            if auto in mid_representation_gold:
                n_corr += 1
        n_gold += len(mid_representation_gold)
        n_pred += len(mid_representation_auto)

    p = 0 if n_pred == 0 else 1. * n_corr / n_pred
    r = 0 if n_gold == 0 else 1. * n_corr / n_gold
    f = 0 if p * r == 0 else 2. * p * r / (p + r)
    print("p = {0}, r = {1}, f = {2}".format(p,r,f))
    return p

if __name__ == '__main__':
    main()