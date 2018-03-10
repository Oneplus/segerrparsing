#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/24 下午1:58
# @Author  : yizhen
# @Site    :
# @File    : cal_pos_f.py
# @Software: PyCharm
from __future__ import print_function
from __future__ import unicode_literals
import codecs
import argparse

ch_punct = {'PU'}
en_punct = {'.', ',', '``', '\'\'', ':'}


def collect(data, head_id, deprel_id, exclude_punct, language):
  start = 0
  result = set()
  mapping = {0: (-1, 0)}
  for i, line in enumerate(data.splitlines()):
    tokens = line.strip().split()
    word = tokens[1]
    mapping[i + 1] = start, len(word)
    start += len(word)

  start = 0
  for i, line in enumerate(data.splitlines()):
    tokens = line.strip().split()
    word = tokens[1]
    if exclude_punct:
      if language == 'ch' and tokens[3] in ch_punct:
        continue
      elif language == 'en' and tokens[3] in en_punct:
        continue
    head_start, head_len = mapping[int(tokens[head_id])]
    result.add((start, len(word), head_start, head_len, tokens[deprel_id]))
    start += len(word)
  return result


def main():
  cmd = argparse.ArgumentParser("Script for evaluating F-score")
  cmd.add_argument('-gold', help="the path to the gold.")
  cmd.add_argument('-auto', help="the path to the prediction.")
  cmd.add_argument('-exclude_punct', action='store_true', default=False, help='exclude punctuation.')
  cmd.add_argument('-language', default='ch', help='used in punctuation.')

  opt = cmd.parse_args()

  with codecs.open(opt.gold, 'r', encoding='utf-8') as fp_gold:
    gold_dataset = fp_gold.read().strip().split('\n\n')

  with codecs.open(opt.auto, 'r', encoding='utf-8') as fp_auto:
    auto_dataset = fp_auto.read().strip().split('\n\n')

  assert len(auto_dataset) == len(gold_dataset)
  n_corr, n_pred, n_gold = 0.0, 0.0, 0.0
  for gold_data, auto_data in zip(gold_dataset, auto_dataset):
    gold_tuples = collect(gold_data, 6, 7, opt.exclude_punct, opt.language)
    auto_tuples = collect(auto_data, 8, 9, opt.exclude_punct, opt.language)
    for gold_tuple in gold_tuples:
      if gold_tuple in auto_tuples:
        n_corr += 1
    n_gold += len(gold_tuples)
    n_pred += len(auto_tuples)

  p = 0 if n_pred == 0 else 1. * n_corr / n_pred
  r = 0 if n_gold == 0 else 1. * n_corr / n_gold
  f = 0 if p * r == 0 else 2. * p * r / (p + r)
  print("p = {0}, r = {1}, f = {2}".format(p, r, f))
  return p


if __name__ == '__main__':
  main()
