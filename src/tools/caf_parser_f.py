#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/31 下午9:52
# @Author  : yizhen
# @Site    : 
# @File    : caf_parser_f.py
# @Software: PyCharm
# 计算一下parser的f值，预测出来的和gold的计算f值，单词，词性和父亲结点和弧关系都对才算一个
import sys
import codecs
import argparse

PU = ['―','!','．','*','半穴式','＂','』','「','《','》','━','、','―――','//','━━','∶','＇','.','＜','……','`','！','）','――','‘',':','，','…','；','：','”','－','」','----','〉','?','──','tw','『','／','-','＞','＊','。','/','・','“',',','’','〈','－－','？','~','33','（','～','PU','＆','―－']

def get_mid_rep(conll, type):
    '''
    conll represented by (start-id_end_id, word, father_id, arc)
    :param conll:
    :return:
    '''
    mid_rep = []
    start_id, end_id = 0, 0

    for word_idx, word in enumerate(conll.strip().split('\n')):
        word = word.strip().split()
        end_id = start_id + len(word[1])
        if type == 'gold':
            mid = (str(start_id)+'_'+str(end_id), word[1], word[6], word[7], word[3])
        elif type == 'auto':
            mid = (str(start_id) + '_' + str(end_id), word[1], word[8], word[9], word[3])
        mid_rep.append(mid)

    return mid_rep

def caf_single_sentence(gold_mid_rep, auto_mid_rep):
    '''
    caf every single sentence
    :param gold_mid_rep:
    :param auto_mid_rep:
    :return:
    '''

    auto_len = len(auto_mid_rep)
    gold_len = len(gold_mid_rep)

    auto_num = auto_len
    gold_num = gold_len

    for i in gold_mid_rep:
        if i[1] in PU:
            gold_num -= 1

    for i in auto_mid_rep:
        if i[1] in PU:
            auto_num -= 1

    n_corr_uas, n_corr_las, index_auto, index_gold = 0, 0, 0, 0
    str_auto, str_gold = "", ""

    while((index_auto < auto_len) and (index_gold < gold_len)):
        auto_word = auto_mid_rep[index_auto][0]
        gold_word = gold_mid_rep[index_gold][0]
        pu_flag = False
        gold_pos = gold_mid_rep[index_gold][4]

        if (auto_word == gold_word):  # 这个是代表单词一致，如果父亲结点也相同，那么就是算对了一个
            if gold_mid_rep[index_gold][1] in PU:  # if in, we can't add it
                pu_flag = True

            auto_father_id = auto_mid_rep[index_auto][2]
            gold_father_id = gold_mid_rep[index_gold][2]

            if auto_mid_rep[int(auto_father_id) - 1][0] == gold_mid_rep[int(gold_father_id) - 1][0]:
                if pu_flag == False:
                    n_corr_uas += 1

                if auto_mid_rep[index_auto][3] == gold_mid_rep[index_gold][3]:
                    if pu_flag == False:
                        n_corr_las += 1


            str_auto += auto_mid_rep[index_auto][1]
            str_gold += gold_mid_rep[index_gold][1]
            index_auto += 1
            index_gold += 1
        else:
            str_auto += auto_mid_rep[index_auto][1]
            str_gold += gold_mid_rep[index_gold][1]
            index_auto += 1
            index_gold += 1

            while index_auto < auto_len and index_gold < gold_len:   # until equal
                if len(str_gold) > len(str_auto):
                    str_auto += auto_mid_rep[index_auto][1]
                    index_auto += 1
                elif len(str_gold) < len(str_auto):
                    str_gold += gold_mid_rep[index_gold][1]
                    index_gold += 1
                else:
                    break
    return n_corr_las, n_corr_uas, auto_num, gold_num


def main():
    parser = argparse.ArgumentParser(description='Evaluation English parsing performance.')
    parser.add_argument('-l', dest='labeled', action='store_true', default=False, help='evaluate LAS')
    parser.add_argument('-d', dest='detailed', action='store_true', default=False,help='log')
    parser.add_argument('gold_conll', help='The path to the filename.')
    parser.add_argument('auto_conll', help='The path to the filename.') 
    args = parser.parse_args()

    auto_conll = args.auto_conll
    gold_conll = args.gold_conll

    #print(auto_conll)
    #print(gold_conll)

    #exit(0)

    n_corr_las, n_corr_uas = 0, 0
    n_pred, n_gold = 0, 0
    p, r, f = 0, 0, 0

    with codecs.open(auto_conll, 'r', encoding='utf-8') as fp_auto:
        fp_auto_conll = fp_auto.read().strip().split('\n\n')
        fp_auto.close()
    fp_auto_conll = sorted(fp_auto_conll, key=lambda d: ''.join([line.split()[1] for line in d.split('\n')]))  # sorted

    with codecs.open(gold_conll, 'r', encoding='utf-8') as fp_gold:
        fp_gold_conll =fp_gold.read().strip().split('\n\n')
        fp_gold.close()

    fp_gold_conll = sorted(fp_gold_conll, key=lambda d: ''.join([line.split()[1] for line in d.split('\n')]))  # sorted

    for sentence_idx, auto_sentence in enumerate(fp_auto_conll):
        gold_sentence = fp_gold_conll[sentence_idx]
        auto_mid_rep = get_mid_rep(auto_sentence, 'auto')
        gold_mid_rep = get_mid_rep(gold_sentence, 'gold')
        single_n_corr_las, single_n_corr_uas, single_n_pred, single_n_gold = caf_single_sentence(gold_mid_rep, auto_mid_rep)
        n_corr_las += single_n_corr_las
        n_corr_uas += single_n_corr_uas
        n_pred += single_n_pred
        n_gold += single_n_gold

    p = 0 if n_pred == 0 else 1. * n_corr_uas / n_pred
    r = 0 if n_gold == 0 else 1. * n_corr_uas / n_gold
    f = 0 if p * r == 0 else 2. * p * r / (p + r)
    #print("p = {0}, r = {1}, f = {2}".format(p, r, f))
    print("{0}".format(f))
if __name__ == '__main__':
    main()
