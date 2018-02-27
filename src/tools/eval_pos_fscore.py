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


def collect(data):
  start = 0
  result = set()
  for line in data.splitlines():
    tokens = line.strip().split()
    word = tokens[1]
    result.add((start, len(word), tokens[3]))
    start += len(word)
  return result


def main():
  cmd = argparse.ArgumentParser("Script for evaluating F-score")
  cmd.add_argument('-gold', help="the path to the gold.")
  cmd.add_argument('-auto', help="the path to the prediction.")

  opt = cmd.parse_args()

  with codecs.open(opt.gold, 'r', encoding='utf-8') as fp_gold:
    gold_dataset = fp_gold.read().strip().split('\n\n')

  with codecs.open(opt.auto, 'r', encoding='utf-8') as fp_auto:
    auto_dataset = fp_auto.read().strip().split('\n\n')

  assert len(auto_dataset) == len(gold_dataset)
  n_corr, n_pred, n_gold = 0.0, 0.0, 0.0
  for gold_data, auto_data in zip(gold_dataset, auto_dataset):
    gold_tuples = collect(gold_data)
    auto_tuples = collect(auto_data)
    for gold_tuple in gold_tuples:
      if gold_tuple in auto_tuples:
        n_corr += 1
    n_gold += len(gold_tuples)
    n_pred += len(gold_tuples)

  p = 0 if n_pred == 0 else 1. * n_corr / n_pred
  r = 0 if n_gold == 0 else 1. * n_corr / n_gold
  f = 0 if p * r == 0 else 2. * p * r / (p + r)
  print("p = {0}, r = {1}, f = {2}".format(p, r, f))
  return p


if __name__ == '__main__':
  main()
