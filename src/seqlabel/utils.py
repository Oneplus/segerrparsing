#!/usr/bin/env python
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


def tag2intervals(tag):
  def tag2id(t):
    # tricky!
    if t.lower() == 'b':
      return 0
    elif t.lower() == 'i':
      return 1
    elif t.lower() == 'e':
      return 2
    else:
      return 3

  if isinstance(tag[0], str) or isinstance(tag[0], unicode):
    tag = [tag2id(t) for t in tag]
  assert isinstance(tag[0], int)
  intervals = []
  start = None
  for i, t in enumerate(tag):
    if t == 0 or t == 3:
      if start is not None:
        intervals.append((start, i - 1))
      start = i
  intervals.append((start, len(tag) - 1))
  return intervals


def f_score(gold, predication):
  assert len(gold) == len(predication)
  n_corr, n_gold, n_pred = 0, 0, 0
  for gold_, pred_ in zip(gold, predication):
    gold_intervals = tag2intervals(gold_)
    pred_intervals = tag2intervals(pred_)
    seg = set()
    for interval in gold_intervals:
      seg.add(interval)
    for interval in pred_intervals:
      if interval in seg:
        n_corr += 1
    n_gold += len(gold_intervals)
    n_pred += len(pred_intervals)
  p = 0 if n_pred == 0 else 1. * n_corr / n_pred
  r = 0 if n_gold == 0 else 1. * n_corr / n_gold
  f = 0 if p * r == 0 else 2. * p * r / (p + r)
  return p, r, f
