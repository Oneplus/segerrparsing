# coding: utf-8
from __future__ import division
import argparse
import random

def split_data(args):
    """
    split 5_fold data
    :param args:
    :return:
    """
    origin_path = args.origin_data_path
    fp_origin = open(origin_path)
    path_directory = args.path

    origin_content = fp_origin.read().strip().split('\n')
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
        str_test = '\n'.join(test_data)
        str_train = '\n'.join(train_data)
        train_path = path_directory+"fold_" + str(i+1) + "/" + name_str_train+".txt"
        test_path = path_directory+"fold_" + str(i+1) + "/" + name_str_test+".txt"
        fp_train = open(train_path, 'w')
        fp_test = open(test_path, 'w')
        fp_train.write(str_train)
        fp_test.write(str_test)
        fp_train.close()
        fp_test.close()

    fp_origin.close()
def main():
    arparser = argparse.ArgumentParser("split data for 5_fold_experiment")
    arparser.add_argument("--origin_data_path", help = "data to be splited", default="../../data/train.txt")
    arparser.add_argument("--path", help = "path directory", default="../../outputs/")

    args = arparser.parse_args()
    split_data(args)

if __name__ == '__main__':
    main()