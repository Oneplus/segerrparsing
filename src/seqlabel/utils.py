#!/usr/bin/env python
from __future__ import unicode_literals
import collections
import itertools


def flatten(lst):
  return list(itertools.chain.from_iterable(lst))


def deep_iter(x):
  if isinstance(x, list) or isinstance(x, tuple):
    for u in x:
      for v in deep_iter(u):
        yield v
  else:
    yield


def dict2namedtuple(dic):
  return collections.namedtuple('Namespace', dic.keys())(**dic)


def tag2intervals(tags, id_to_tag):
  if isinstance(tags[0], int):
    tags = [id_to_tag.get(t) for t in tags]
  assert isinstance(tags[0], str) or isinstance(tags[0], unicode)
  intervals = set()
  start = None
  for i, t in enumerate(tags):
    if t.lower() == 'b' or t.lower() == 's':
      if start is not None:
        intervals.add((start, i - 1))
      start = i
  intervals.add((start, len(tags) - 1))
  return intervals


def f_score(gold, prediction, id_to_tag=None):
  assert len(gold) == len(prediction)

  if id_to_tag is None:
    id_to_tag = {0: 'b', 1: 'i', 2: 'e', 3: 's'}
  n_corr, n_gold, n_pred = 0, 0, 0
  for gold_, pred_ in zip(gold, prediction):
    gold_intervals = tag2intervals(gold_, id_to_tag)
    pred_intervals = tag2intervals(pred_, id_to_tag)
    for gold_interval in gold_intervals:
      if gold_interval in pred_intervals:
        n_corr += 1
    n_gold += len(gold_intervals)
    n_pred += len(pred_intervals)
  p = 0 if n_pred == 0 else 1. * n_corr / n_pred
  r = 0 if n_gold == 0 else 1. * n_corr / n_gold
  f = 0 if p * r == 0 else 2. * p * r / (p + r)
  return p, r, f
