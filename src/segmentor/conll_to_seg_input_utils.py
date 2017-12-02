# coding: utf-8
# convert the conll format to the format used as the segmentor input
import argparse

def data_every_conll_utils(sentence):
    """
    process every sentence
    :param sentence:  sentence's conll format
    :return:   type:str
    """
    tag = []
    character = []
    res = ""

    sentence_list = sentence.strip().split('\n')
    for index, conll_line in enumerate(sentence_list):
        conll_line = conll_line.strip().split()
        word = conll_line[1]
        for char in word:
            character.append(char)
        if len(word) == 1:
            tag.append('S')
        elif len(word) == 2:
            tag.append('B')
            tag.append('E')
        else:
            I_count = len(word) - 2
            tag.append('B')
            for i in range(I_count):
                tag.append('I')
            tag.append('E')
    for index, char in enumerate(character):
        res += char
        if index != len(character)-1:
            res += ' '
    res += '\t'
    for index, tagg in enumerate(tag):
        res += tagg
        if index != len(tag) - 1:
            res += ' '
    return res


def data_utils(args):
    """
    :param args:
    :return:
    """
    conll_path = args.conll_path
    type = ""
    if conll_path.find("test") != -1:  #  not equal -1 represents finding the substring "test"
        type = "test.txt"
    elif conll_path.find("train") != -1:
        type = "train.txt"
    else:
        type = "valid.txt"

    output_path = args.output_path

    # 从右往左走找到第一个/
    for i in range(len(output_path))[::-1]:
        if output_path[i] == '/':
            break
    index = i
    output_path = output_path[:index+1]+type

    output_path = output_path.replace('data', 'outputs')

    fp_conll = open(conll_path)
    fp_output = open(output_path, 'w')

    conll = fp_conll.read().strip().split('\n\n')
    res_output = ""

    for index, sentence in enumerate(conll):
        res_every_conll = data_every_conll_utils(sentence)
        res_output +=  res_every_conll
        res_output += '\n'

    fp_output.write(res_output)

    fp_conll.close()
    fp_output.close()

def main():
    argparser = argparse.ArgumentParser("conll_to_seg_input_utils")
    argparser.add_argument('--conll_path', default='../../data/CTB5.1-devel.gp.conll')
    argparser.add_argument('--output_path', default='../../data/output_seg.txt')

    args = argparser.parse_args()
    data_utils(args)

if __name__ == '__main__':
    main()
