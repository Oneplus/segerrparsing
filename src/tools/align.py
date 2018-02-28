#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import sys
import argparse
import codecs
unk = '<UNK>'


class CoNLLLine(object):
  def __init__(self, id, form, postag, head, deprel, start, end):
    self.id = id
    self.form = form
    self.postag = postag
    self.head = head
    self.deprel = deprel
    self.start = start
    self.end = end


def print_one_line(entry, handler):
  print('{id}\t{form}\t{form}\t{pos}\t{pos}\t{pos}\t{head}\t{deprel}\t_\t_'.format(
    id=entry.id, form=entry.form, pos=entry.postag, head=entry.head, deprel=entry.deprel), file=handler)


def align_one_instance(gold_data, auto_data):
  inst = [line.split() for line in gold_data.splitlines()]
  start_id = 0
  new_inst = []
  for token in inst:
    if len(token) == 6:
      token.insert(1, '\u3000')
      token.insert(2, '\u3000')
    end_id = start_id + len(token[1])
    new_inst.append(CoNLLLine(id=token[0], form=token[1], postag=token[3], head=token[6], deprel=token[7],
                              start=start_id, end=end_id))
    start_id = end_id
  gold_inst = new_inst

  inst = [line.split() for line in auto_data.splitlines()]
  new_inst = []
  start_id = 0
  for i, token in enumerate(inst):
    if len(token) == 6:
      token.insert(1, '\u3000')
      token.insert(2, '\u3000')
    end_id = start_id + len(token[1])
    new_inst.append(CoNLLLine(id=str(i + 1), form=token[1], postag=None, head=None, deprel=None,
                              start=start_id, end=end_id))
    start_id = end_id
  auto_inst = new_inst

  gold_len, auto_len = len(gold_inst), len(auto_inst)
  gold_i, auto_i = 0, 0
  gold, auto = "", ""

  auto_to_gold_alignment = {}
  gold_to_auto_alignment = {}
  while auto_i < auto_len and gold_i < gold_len:
    gold_entry, auto_entry = gold_inst[gold_i], auto_inst[auto_i]
    gold_word, auto_word = gold_entry.form, auto_entry.form

    if auto_word == gold_word:
      auto_to_gold_alignment[auto_entry.start, auto_entry.end] = gold_entry.start, gold_entry.end
      gold_to_auto_alignment[gold_entry.start, gold_entry.end] = auto_entry.start, auto_entry.end

      gold += gold_word
      auto += auto_word
      gold_i += 1
      auto_i += 1
    else:
      gold += gold_word
      auto += auto_word
      gold_i += 1
      auto_i += 1
      while gold_i < gold_len and auto_i < auto_len:
        if len(gold) > len(auto):
          auto += auto_inst[auto_i].form
          auto_i += 1
        elif len(gold) < len(auto):
          gold += gold_inst[gold_i].form
          gold_i += 1
        else:
          break

  gold_inst_by_position = {(entry.start, entry.end): entry for entry in gold_inst}
  auto_inst_by_position = {(entry.start, entry.end): entry for entry in auto_inst}
  gold_inst_by_index = {entry.id: entry for entry in gold_inst}
  for entry in auto_inst:
    if (entry.start, entry.end) not in auto_to_gold_alignment:
      entry.postag = unk
      entry.head = unk
      entry.deprel = unk
    else:
      gold_start, gold_end = auto_to_gold_alignment[entry.start, entry.end]
      gold_entry = gold_inst_by_position[gold_start, gold_end]
      entry.postag = gold_entry.postag
      if gold_entry.head == '0':
        entry.head = '0'
        entry.deprel = gold_entry.deprel
      else:
        gold_parent_entry = gold_inst_by_index[gold_entry.head]
        if (gold_parent_entry.start, gold_parent_entry.end) in gold_to_auto_alignment:
          auto_parent_entry = auto_inst_by_position[
            gold_to_auto_alignment[gold_parent_entry.start, gold_parent_entry.end]]
          entry.head = auto_parent_entry.id
          entry.deprel = gold_entry.deprel
        else:
          entry.head = unk
          entry.deprel = unk

  return auto_inst


def align(gold_data, auto_data, output):
  if output is not None:
    handler = codecs.open(output, 'w', encoding='utf-8')
  else:
    handler = codecs.getwriter('utf-8')(sys.stdout)
  for gold_inst, auto_inst in zip(gold_data, auto_data):
    new_inst = align_one_instance(gold_inst, auto_inst)
    for entry in new_inst:
      print_one_line(entry, handler)
    print('', file=handler)


if __name__ == '__main__':
  cmd = argparse.ArgumentParser('align auto sentence and gold sentence')
  cmd.add_argument("--gold", help="path of gold conll")
  cmd.add_argument("--auto", help="path of aligned sentence")
  cmd.add_argument("--output", help="path of auto sentence")
  opt = cmd.parse_args()

  gold_data = codecs.open(opt.gold, 'r').read().strip().split('\n\n')
  auto_data = codecs.open(opt.auto, 'r').read().strip().split('\n\n')

  align(gold_data, auto_data, opt.output)
