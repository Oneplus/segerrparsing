#!/usr/bin/env python
# @Author  : yizhen
# @Site    : 
# @File    : add_postag.py.py
# @Software: PyCharm
from __future__ import print_function
from __future__ import unicode_literals
import codecs
import argparse


def main():
  cmd = argparse.ArgumentParser("add postag")
  cmd.add_argument("-gold", help="the path to the gold file.")
  cmd.add_argument("-auto", help="the path to the auto file.")
  cmd.add_argument("-output", help="the target output")

  opt = cmd.parse_args()
  gold_dataset = codecs.open(opt.gold, 'r', encoding='utf-8').read().strip().split('\n\n')
  auto_dataset = codecs.open(opt.auto, 'r', encoding='utf-8').read().strip().split('\n\n')
  assert len(gold_dataset) == len(auto_dataset)

  with codecs.open(opt.output, 'w', encoding='utf-8') as fpo:
    for gold_data, auto_data in zip(gold_dataset, auto_dataset):
      gold_lines = gold_data.splitlines()
      auto_lines = auto_data.splitlines()
      assert len(gold_lines) == len(auto_lines)

      for gold_line, auto_line in zip(gold_lines, auto_lines):
        gold_tokens = gold_line.split()
        auto_tokens = auto_line.split()
        assert len(gold_tokens) >= 8
        if len(gold_tokens) == 8:
          gold_tokens.extend(['_', '_'])
        gold_tokens[3], gold_tokens[4] = auto_tokens[3], auto_tokens[4]
        print('\t'.join(gold_tokens), file=fpo)
      print(file=fpo)


if __name__ == '__main__':
    main()
