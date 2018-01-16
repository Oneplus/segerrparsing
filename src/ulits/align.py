#!/usr/bin/env python
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


def load_conll_data(path, skip_abnormal=True):
  # exceptions:
  # - one gold sentence has different segmentation
  # - one gold sentence has one segmentation, but different postag or dependencies
  output = {}
  with codecs.open(path, 'r', encoding='utf-8') as f:
    dataset = f.read().strip().split('\n\n')
    for data in dataset:
      lines = data.split('\n')
      abnormal = any([len(line.split()) != 8 for line in lines])
      if abnormal:
        print('ABNORMAL:\n{0}'.format(data), file=sys.stderr)
        if skip_abnormal:
          continue
      key = ''.join([line.split()[1] for line in lines])
      if key in output:
        output[key].append(data)
      else:
        output[key] = [data]
  return output


def load_segment_data(path):
  output = {}
  with codecs.open(path, 'r', encoding='utf-8') as f:
    for line in f:
      data = line.strip()
      key = ''.join(data.split())
      if key in output:
        output[key].append(data)
      else:
        output[key] = [data]
  return output


def sanity_check(conll_data, segment_data):
  conll_keys = conll_data.keys()
  segment_keys = segment_data.keys()
  # print(list(set(conll_keys) - set(segment_keys)))
  # print(list(set(segment_keys) - set(conll_keys)))
  assert (set(conll_keys) & set(segment_keys)) == set(conll_keys)

  for key in conll_keys:
    assert len(conll_data[key]) == len(segment_data[key])
    for inst in conll_data[key]:
      lines = inst.splitlines()
      for i, line in enumerate(lines):
        tokens = line.split()
        assert len(tokens) == 8, '{0}, {1}'.format(inst, key)
        assert int(tokens[0]) == i + 1


def print_one_line(entry, handler):
  print('{id}\t{form}\t{form}\t{pos}\t{pos}\t_\t{head}\t{deprel}\t_\t_'.format(
    id=entry.id, form=entry.form, pos=entry.postag, head=entry.head, deprel=entry.deprel), file=handler)


def align_one_instance(conll_inst, segment_inst):
  inst = [line.split() for line in conll_inst.splitlines()]
  start_id = 0
  new_inst = []
  for token in inst:
    end_id = start_id + len(token[1])
    new_inst.append(CoNLLLine(id=token[0], form=token[1], postag=token[3], head=token[6], deprel=token[7],
                              start=start_id, end=end_id))
    start_id = end_id
  gold_inst = new_inst

  inst = segment_inst.split()
  new_inst = []
  start_id = 0
  for i, token in enumerate(inst):
    end_id = start_id + len(token)
    new_inst.append(CoNLLLine(id=str(i + 1), form=token, postag=None, head=None, deprel=None,
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


def align(conll_data, segment_data, output):
  keys = conll_data.keys()
  handler = codecs.open(output, 'w', encoding='utf-8')
  for key in keys:
    conll_instances = conll_data[key]
    segment_instances = segment_data[key]
    for conll_inst, segment_inst in zip(conll_instances, segment_instances):
      new_inst = align_one_instance(conll_inst, segment_inst)
      for entry in new_inst:
        print_one_line(entry, handler)
      print('', file=handler)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('align auto sentence and gold sentence')
    cmd.add_argument("--gold_conll", help="path of gold conll",default='../../data/conll/CTB5.1-devel.gp.conll')
    cmd.add_argument("--auto_seg", help="path of aligned sentence",default='../../data/conll/auto_devel.sentence')
    cmd.add_argument("--output", help="path of auto sentence",default='../../data/conll/output.conll')
    args = cmd.parse_args()

    conll_data = load_conll_data(args.gold_conll)
    autoseg_data = load_segment_data(args.auto_seg)

    sanity_check(conll_data, autoseg_data)
    align(conll_data, autoseg_data, args.output)
