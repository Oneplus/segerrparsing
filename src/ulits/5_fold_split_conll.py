# coding: utf-8
from __future__ import division
import argparse
import random
import codecs

def split_data(args):
    """
    split 5_fold data
    :param args:
    :return:
    """
    origin_path = args.origin_data_path
    fp_origin = codecs.open(origin_path, 'r', encoding='utf-8')
    path_directory = args.path

    origin_content = fp_origin.read().strip().split('\n\n')
    number = len(origin_content)

    shuffle_content = random.shuffle(origin_content)
    name_str_train = "train"
    name_str_test = "test"
    # print(number)
    interval1 = int(number) // 5
    for i in range(5):
        if i == 4:
            end_interval = number
        else:
            end_interval = (i + 1) * interval1
        test_data = origin_content[i*interval1:end_interval]
        train_data = origin_content[0:(i*interval1)]+origin_content[end_interval:number]
        str_test = '\n\n'.join(test_data)
        str_train = '\n\n'.join(train_data)
        train_path = path_directory+"fold_" + str(i+1) + "/" + name_str_train+".txt"
        test_path = path_directory+"fold_" + str(i+1) + "/" + name_str_test+".txt"
        fp_train = codecs.open(train_path, 'w', encoding='utf-8')
        fp_test = codecs.open(test_path, 'w', encoding='utf-8')
        fp_train.write(str_train)
        fp_test.write(str_test)
        fp_train.close()
        fp_test.close()

    fp_origin.close()
def main():
    arparser = argparse.ArgumentParser("split data for 5_fold_conll experiment")
    arparser.add_argument("--origin_data_path", help = "data to be splited", default="../../data/CTB5.1/CTB5.1-train_test.gp.conll")
    arparser.add_argument("--path", help = "path directory", default="../../data/CTB5.1/test/")

    args = arparser.parse_args()
    split_data(args)

if __name__ == '__main__':
    main()