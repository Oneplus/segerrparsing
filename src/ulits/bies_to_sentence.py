# -*- coding:utf-8 -*-
from __future__ import print_function
import argparse
import codecs


def bies_to_segmentation(word, tags):
    """
    convert every sentence
    :param word:  str
    :param tags:  str
    :return: str: sentence
    """
    word = word.split()
    tags = tags.split()
    res = []
    single_word = ""
    for index_tag, tag in enumerate(tags):
        if tag == 'S':
            res.append(word[index_tag])
        elif tag == 'B':  # B represents the begin of a word
            single_word += word[index_tag]
        elif tag == 'I':
            single_word += word[index_tag]
        elif tag == 'E':  # E represents the end of a word
            single_word += word[index_tag]
            res.append(single_word)
            single_word = ""
    return ' '.join(res)


def main():
    cmd = argparse.ArgumentParser("convert the tag representation to sentence")
    cmd.add_argument("--input", help="the path of tag representation", default='../data/test_res.txt')
    cmd.add_argument("--output", help="the path of sentence representation", default='../data/sentence.txt')

    args = cmd.parse_args()
    fpo = codecs.open(args.output, 'w', encoding='utf-8')

    for line in codes.open(args.input, 'r', encoding='utf-8'):
        chars, tags = line.strip().split('\t')
        print(bies_to_segmentation(chars, tags), file=fpo)


if __name__ == '__main__':
    main()
