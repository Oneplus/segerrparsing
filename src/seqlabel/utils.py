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

  if isinstance(tag[0], str):
    tag = [tag2id(t) for t in tag]
  intervals = []
  l = len(tag)
  i = 0
  while i < l:
    if tag[i] == 2 or tag[i] == 3:
      intervals.append((i, i))
      i += 1
      continue
    j = i + 1
    while True:
      if j == l or tag[j] == 0 or tag[j] == 3:
        intervals.append((i, j - 1))
        i = j
        break
      elif tag[j] == 2:
        intervals.append((i, j))
        i = j + 1
        break
      else:
        j += 1
  return intervals


def f_score(gold, predication):
  assert len(gold) == len(predication)
  tp, fp, fn = 0, 0, 0,
  for gold_, pred_ in zip(gold, predication):
    gold_intervals = tag2intervals(gold_)
    pred_intervals = tag2intervals(pred_)
    seg = set()
    for interval in gold_intervals:
      seg.add(interval)
      fn += 1
    for interval in pred_intervals:
      if interval in seg:
        tp += 1
        fn -= 1
      else:
        fp += 1
  p = 0 if tp == 0 else 1. * tp / (tp + fp)
  r = 0 if tp == 0 else 1. * tp / (tp + fn)
  f = 0 if p * r == 0 else 2. * p * r / (p + r)
  return p, r, f
