import argparse
import codecs

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
        flag = False
        sentences_conll = ""
        sentences_auto = ""
        for word_index, word in enumerate(sentence.split('\n')):
            word = word.strip().split()
            sentences_conll += word[1]

        for index_auto, sentences_auto in enumerate(auto_txt_align):
            sentence_auto = ""
            for index, value in enumerate(sentences_auto.split()):
                sentence_auto += value
            # sentence_auto = "".join(auto_txt_align[index_auto].split())
            #print("sentences_conll = {0}, sentences_auto = {1}".format(sentences_conll, sentence_auto))
            # if sentences_conll == sentence_auto:  # find the sentences
            #     if index_auto not in index_list:  # not in it and transit
            #         count += 1
            #         print("has aligned {0} sentences:".format(count))
            #         res_sentence.append(sentence_auto)
            #         index_list.append(index_auto)
            #         break  # one one correspond
            # print("{0}".format(sentences_conll))
            # print("{0}".format(sentence_auto))
            if sentences_conll == sentence_auto:
                flag = True
                count += 1
                print("has aligned {0} sentences:".format(count))
                res_sentence.append(sentence_auto)
                break
        if  flag == False:
            # print("{0} gold conll don't find".format(sentences_conll))
            pass
    res_sentence = '\n'.join(res_sentence)
    fp_auto_temp.write(res_sentence)

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('align auto sentence and gold sentence')
    cmd.add_argument("--gold_conll_path", help = "path of gold conll", default='../../data/CTB5.1/CTB5.1-train.gp.conll')
    cmd.add_argument("--aligned_sentence_path", help = "path of auto sentence", default='../../outputs/aligned_fold_all.txt')
    cmd.add_argument("--auto_sentence_path", help = "path of aligned sentence", default='../../outputs/fold_all.txt')
    args = cmd.parse_args()

    fp_gold_conll = args.gold_conll_path
    fp_fp_auto_temp =args.auto_sentence_path
    fp_auto_txt_align = args.aligned_sentence_path


    f_gold_conll = codecs.open(fp_gold_conll, 'r', encoding='utf-8')
    f_auto_temp = codecs.open(fp_fp_auto_temp, 'r', encoding='utf-8')
    f_auto_txt_align = codecs.open(fp_auto_txt_align, 'w', encoding='utf-8')

    gold_conll = f_gold_conll.read().strip().split('\n\n')
    auto_txt_align = f_auto_temp.read().strip().split('\n')

    align(f_auto_txt_align, gold_conll, auto_txt_align)





