# -*- coding:utf-8 -*-
import argparse
import codecs
def is_in_auto(auto_mid_represent, gold_father_represent):
    """
    judge father represent is whether in auto_seg_list
    if yes return father_id
    :param auto_mid_represent:
    :param gold_father_represent:
    :return:
    """
    flag = False
    for index, value in enumerate(auto_mid_represent):
        if auto_mid_represent[index][0] == gold_father_represent[0] and auto_mid_represent[index][1] == gold_father_represent[1]:
            flag = True  # find it
            return index
    # not in ,return -1
    return -1

def convert_to_conll(args, auto_mid_represent):
    """
    convert (word, start_id_end, arc, father, id) to conll format
    :param auto_mid_represent:
    :return:
    """
    str_one_sentence = ""
    for index, value in enumerate(auto_mid_represent):
        str_one_sentence += str(index+1)
        str_one_sentence += '\t'
        str_one_sentence += value[0]
        str_one_sentence += '\t'
        str_one_sentence += value[0]
        str_one_sentence += '\t'
        str_one_sentence += 'CIXIN\t'
        str_one_sentence += 'CIXIN\t'
        str_one_sentence += 'CIXIN\t'
        str_one_sentence += str(value[3])
        str_one_sentence += '\t'
        str_one_sentence += value[2]
        str_one_sentence += '\n'

    return str_one_sentence

def transit(args, auto_mid_represent, gold_mid_represent):
    """

    :param auto_mid_represent:
    :param gold_mid_represent:
    :return:
    """
    len_gold = len(gold_mid_represent)
    len_auto = len(auto_mid_represent)

    index_gold, index_auto = 0,0
    str_auto, str_gold = "", ""

    while index_auto < len_auto and index_gold < len_gold:
        auto_word = auto_mid_represent[index_auto][0]
        gold_word = gold_mid_represent[index_gold][0]

        if auto_word == gold_word:  # 如果相等就有可能需要重新标注，看gold的父亲是否在auto中出现
            father_id = gold_mid_represent[index_gold][3]
            father_list = gold_mid_represent[int(father_id)]

            id = is_in_auto(auto_mid_represent, father_list)
            if id != -1:  # 如果在里面
                auto_mid_represent[index_auto][3] = id
                auto_mid_represent[index_auto][2] = gold_mid_represent[index_gold][2]
                str_auto += auto_mid_represent[index_auto][0]
                str_gold += gold_mid_represent[index_gold][0]

        if len(str_auto) < len(str_gold):
            index_auto += 1
            str_auto += auto_mid_represent[index_auto][0]
        elif len(str_auto) > len(str_gold):
            index_gold += 1
            str_gold += gold_mid_represent[index_gold][0]
        else:
            index_gold += 1
            str_gold = ""
            index_auto += 1
            str_auto = ""

    #print(auto_mid_represent)

    return convert_to_conll(args, auto_mid_represent)


def transit_one_sentence(args, auto_sentence, gold_conll_sentence):
    """
    :param auto_sentence:
    :param gold_conll_sentence:
    :return:
    """

    auto_sentence = auto_sentence.strip().split()
    gold_conll_sentence = gold_conll_sentence.strip().split('\n')
    auto_mid_represent = []
    gold_mid_represent = []
    start_id, end_id = 0, 0

    # represent every word of auto seg to this format [word start_id_end_id arc father_id]
    for index, value in enumerate(auto_sentence):
        temp = []
        end_id = start_id + len(value)
        id = str(start_id)+"_"+ str(end_id)
        temp = [value] + [id] + ["AUTO_HEAD"] + ["-0"]+[index]
        auto_mid_represent.append(temp)
        start_id = start_id+len(value)

    start_id, end_id = 0, 0

    # represent every word of gold seg to this format [word start_id_end_id arc father_id]
    for index, value in enumerate(gold_conll_sentence):
        value = value.strip().split()
        temp = []
        end_id = start_id + len(value[1])
        id = str(start_id) + "_" + str(end_id)
        temp = [value[1]] + [id] + [value[7]] + [value[6]] + [index]
        gold_mid_represent.append(temp)
        start_id = start_id + len(value[1])

    return transit(args, auto_mid_represent, gold_mid_represent)

def align(fp_auto_temp, gold_conll, auto_txt_align):
    '''
    the result restore in fp_auto_temp
    :param fp_auto_temp:
    :param gold_conll:
    :param auto_txt_align:
    :return:
    '''
    count = 0
    res_sentence = []
    index_list = []  # find it and delete it
    for sentence_index, sentence in enumerate(gold_conll):
        # print("align {0}".format(sentence_index))
        sentences_conll = ""
        sentences_auto = ""
        for word_index, word in enumerate(sentence.split('\n')):
            word = word.strip().split()
            sentences_conll += word[1]

        for index_auto, sentences_auto in enumerate(auto_txt_align):
            sentence_auto = "".join(auto_txt_align[index_auto].split())
            #print("sentences_conll = {0}, sentences_auto = {1}".format(sentences_conll, sentence_auto))
            if sentences_conll == sentence_auto:  # find the sentences
                if index_auto not in index_list:  # not in it and transit
                    count += 1
                    print("has aligned {0} sentences:".format(count))
                    res_sentence.append(sentence_auto)
                    index_list.append(index_auto)
                    break  # one one correspond

    res_sentence = '\n'.join(res_sentence)
    fp_auto_temp.write(res_sentence)


def transited(args):
    gold_conll_path  =args.gold_conll_path
    auto_conll_path = args.auto_conll_path
    # auto_seg_txt = args.auto_seg_txt
    auto_seg_align = args.auto_seg_adjust_order

    fp_gold = codecs.open(gold_conll_path, 'r', encoding='utf-8')

    fp_auto_align = codecs.open(auto_seg_align, 'r', encoding='utf-8')
    fp_auto = codecs.open(auto_conll_path, 'w', encoding='utf-8')
    # fp_auto_temp = codecs.open(auto_seg_txt, 'w', encoding='utf-8')
    fp_auto_txt = codecs.open(auto_seg_txt, 'r', encoding='utf-8')

    auto_txt_align = fp_auto_align.read().strip().split('\n')


    gold_conll = fp_gold.read().strip().split('\n\n')
    # align(fp_auto_temp, gold_conll, auto_txt_align)  # auto_txt is the aligned order
    auto_txt = fp_auto_txt.read().strip().split('\n')

    res_trainsit = ""
    for index_sentence, sentence in enumerate(auto_txt):
        print(index_sentence)
        transit_sentence = transit_one_sentence(args, sentence, gold_conll[index_sentence])
        res_trainsit += transit_sentence
        res_trainsit += '\n'

    fp_auto.write(res_trainsit)

    fp_auto_txt.close()
    fp_gold.close()
    fp_auto.close()

def main():
    cmd = argparse.ArgumentParser('transit gold seg to auto seg format conll')
    cmd.add_argument("--gold_conll_path", help = "the path of gold conll", default='../../data/CTB5.1/CTB5.1-train.gp.conll')
    cmd.add_argument("--auto_seg_adjust_order", help = "the path of auto seg", default='../../data/CTB5.1/test/predict_train.txt')
    # cmd.add_argument("--auto_seg_txt", help="auto seg after align order", default='../../data/CTB5.1/test/auto_txt_align_train.txt')
    cmd.add_argument("--auto_conll_path", help = "the path of auto conll", default='../../data/CTB5.1/test/auto.conll')

    args = cmd.parse_args()

    transited(args)

if __name__ == '__main__':
    main()