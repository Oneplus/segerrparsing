# coding:utf-8
import argparse

def convert_every_sentence(word, tags):
    """
    convert every sentence
    :param word:  str
    :param tags:  str
    :return: str: sentence
    """
    word = word.split()
    tags = tags.split()
    res = []
    str_res = ""
    single_word = ""
    for index_tag, tag in enumerate(tags):
        if tag == 'S':
            res.append(word[index_tag])
        elif tag == 'B':  # B represents the begin of a word
            single_word+=word[index_tag]
        elif tag == 'I':
            single_word+=word[index_tag]
        elif tag == 'E':  # E represents the end of a word
            single_word += word[index_tag]
            res.append(single_word)
            single_word = ""
    str_res = ' '.join(res)
    str_res += '\n'
    return str_res

def convert_function(args):
    """
    :param args:
    :return:
    """
    res_sentence = ""
    tag_path = args.tag_path
    sentence_path =args.sentence_path
    fp_resources = open(tag_path, encoding='utf-8')
    fp_target = open(sentence_path, 'w', encoding='utf-8')

    resources = fp_resources.read().strip().split('\n')
    for index, value in enumerate(resources):
        word, tags = value.split('\t')
        res_sentence += convert_every_sentence(word, tags)

    fp_target.write(res_sentence)

    fp_resources.close()
    fp_target.close()

def main():
    arparser = argparse.ArgumentParser("convert the tag representation to sentence")
    arparser.add_argument("--tag_path", help = "the path of tag representation", default='../data/test_res.txt')
    arparser.add_argument("--sentence_path", help = "the path of sentence representation", default='../data/sentence.txt')

    args = arparser.parse_args()
    convert_function(args)

if __name__ == '__main__':
    main()