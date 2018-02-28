# coding: utf-8
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import argparse
import random
import codecs


def main():
  cmd = argparse.ArgumentParser("split data for 5_fold_conll experiment")
  cmd.add_argument("--input", help="data to be splited")
  cmd.add_argument("--output", help="path directory")
  cmd.add_argument("--nfold", type=int, default=5, help="the number of folds")

  opt = cmd.parse_args()

  random.seed(1234)
  dataset = codecs.open(opt.input, 'r', encoding='utf-8').read().strip().split('\n\n')
  n_sent = len(dataset)
  random.shuffle(dataset)

  # print(number)
  fold_size = ((n_sent // opt.nfold) + (1 if n_sent % opt.nfold != 0 else 0))
  basename = os.path.basename(opt.input)
  for i in range(opt.nfold):
    start, end = i * fold_size, min((i + 1) * fold_size, n_sent)
    train_path = os.path.join(opt.output, '{0}.{1}.train'.format(basename, i))
    test_path = os.path.join(opt.output, '{0}.{1}.test'.format(basename, i))

    with codecs.open(train_path, 'w', encoding='utf-8') as fpo:
      print('\n\n'.join([data for k, data in enumerate(dataset) if k < start or k >= end]), end='\n', file=fpo)

    with codecs.open(test_path, 'w', encoding='utf-8') as fpo:
      print('\n\n'.join([data for k, data in enumerate(dataset) if start <= k < end]), end='\n', file=fpo)


if __name__ == '__main__':
    main()
