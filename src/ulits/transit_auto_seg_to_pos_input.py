#!/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import argparse

def is_in_conll(value_word, id, gold_mid_represent):
    '''
    judge word whether is in gold_mid_represent
    :param value_word:
    :param id:
    :param gold_mid_represent:
    :return:
    '''
    idx = -1

    for index, value in enumerate(gold_mid_represent):
        if value[0] == value_word and value[1] == id:
            idx = index  # find it
            break

    return idx

def transit_one_sentence(auto_seg, conll_sentnece):
    '''
    transit one sentence
    :param auto_seg:
    :param conll_sentnece:
    :return:
    '''
    res_one_sentence_pos = []
    list_seg = auto_seg.split()
    line_conll = conll_sentnece.split('\n')

    index_seg, index_conll = 0, 0
    str_seg, str_conll = "", ""


    start_id, end_id = 0, 0
    gold_mid_represent = []

    # represent every word of gold seg to this format [word start_id_end_id arc father_id current_id pos]
    for index, value in enumerate(line_conll):
        value = value.strip().split()
        temp = []
        end_id = start_id + len(value[1])
        id = str(start_id) + "_" + str(end_id)
        temp = [value[1]] + [id] + [value[7]] + [value[6]] + [index] + [value[3]]
        gold_mid_represent.append(temp)
        start_id = start_id + len(value[1])

    start_id, end_id = 0, 0
    for index_word, value_word in enumerate(list_seg):  # traverse every word
        end_id = start_id + len(value_word)
        id = str(start_id) + "_" + str(end_id)
        idx = is_in_conll(value_word, id, gold_mid_represent)
        if idx != -1:  # find it
            temp = ""
            temp += value_word
            temp += '_'
            temp += gold_mid_represent[idx][5]

        else:
            temp = ""
            temp += value_word
            temp += '_'
            temp += 'CIXIN'
        res_one_sentence_pos.append(temp)
        start_id = start_id + len(value_word)

    # print(res_pos)
    return ' '.join(res_one_sentence_pos)

def judge_illegal(conll):
    '''
    judge the conll whether is illegal
    :param conll:
    :return:
    '''
    conll = conll.strip().split('\n')
    print(conll)
    for index, value in enumerate(conll):

        if(len(value.strip().split()) != 8):
            return False
    return True

def transit(args):
    '''
    transit auto seg format to partial pos input
    :param args:
    :return:
    '''
    gold_conll_path = args.gold_conll_path
    auto_seg_path = args.auto_seg_path
    partial_pos_res = args.partial_pos_res

    fp_gold_conll = codecs.open(gold_conll_path, 'r', encoding='utf-8')
    fp_auto_seg = codecs.open(auto_seg_path, 'r', encoding='utf-8')
    fp_partial_pos_res = codecs.open(partial_pos_res, 'w', encoding='utf-8')
    res_pos = []

    content_auto_seg = fp_auto_seg.read().strip().split('\n')
    content_gold_conll = fp_gold_conll.read().strip().split('\n\n')

    for index_auto_seg, auto_seg in enumerate(content_auto_seg):
        # 需要判断一下是否有conll中的某行少于一定列数，如果是，就continue
        # print("auto_seg:{0}".format(auto_seg))
        if judge_illegal(content_gold_conll[index_auto_seg]) == False:
            continue
        res_pos.append(transit_one_sentence(auto_seg, content_gold_conll[index_auto_seg]))
    res_pos = '\n'.join(res_pos)

    fp_partial_pos_res.write(res_pos)

    fp_gold_conll.close()
    fp_auto_seg.close()
    fp_partial_pos_res.close()


def main():
    cmd = argparse.ArgumentParser("transit auto seg to pos input")
    cmd.add_argument("--gold_conll_path", help = "path of gold conll", default='../../data/pos/test.conll')
    cmd.add_argument("--auto_seg_path", help="path of auto seg", default='../../data/pos/aligned.txt')
    cmd.add_argument("--partial_pos_res", help="path of partial pos res", default='../../data/pos/res_pos.txt')

    args = cmd.parse_args()
    transit(args)

if __name__ == '__main__':
    main()