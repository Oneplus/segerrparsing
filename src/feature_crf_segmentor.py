#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse
import codecs
import pycrfsuite
from seqlabel.utils import f_score


# copied from CRFsuite example
def word2features(sent, i):
  ch = sent[i][0]
  prev_ch = sent[i - 1][0] if i > 0 else u'<BOS>'
  next_ch = sent[i + 1][0] if i < len(sent) - 1 else u'<EOS>'
  prev_ch2 = sent[i - 2][0] if i > 1 else u'<BOS>'
  next_ch2 = sent[i + 2][0] if i < len(sent) - 2 else u'<EOS>'
  features = [
    u'c0={0}'.format(ch),
    u'c-1={0}'.format(prev_ch),
    u'c-2={0}'.format(prev_ch2),
    u'c+1={0}'.format(next_ch),
    u'c+2={0}'.format(next_ch2),
    u'c-2={0}|c-1={1}'.format(prev_ch2, prev_ch),
    u'c-1={0}|c0={1}'.format(prev_ch, ch),
    u'c0={0}|c+1={1}'.format(ch, next_ch),
    u'c+1={0}|c+2={1}'.format(next_ch, next_ch2),
    u'c-2={0}|c-1={1}|c0={2}'.format(prev_ch2, prev_ch, ch),
    u'c-1={0}|c0={1}|c+1={2}'.format(prev_ch, ch, next_ch),
    u'c0={0}|c+1={1}|c+2={2}'.format(ch, next_ch, next_ch2),
  ]

  return features


def sent2features(sent):
  return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
  return [label for token, label in sent]


def sent2tokens(sent):
  return [token for token, label in sent]


def load_corpus(filename):
  sents = []
  for line in codecs.open(filename, 'r', encoding='utf-8'):
    fields = line.strip().split(u'\t')
    assert len(fields) == 2
    chars = fields[0].split()
    tags = fields[1].split()
    assert len(chars) == len(tags)
    sents.append(zip(chars, tags))
  return sents


def main():
  cmd = argparse.ArgumentParser('A feature CRF baseline for word segmentation.')
  cmd.add_argument('--train', required=True, help='the path to the training file.')
  cmd.add_argument('--devel', required=True, help='the path to the development file.')
  cmd.add_argument('--output', help='the path to the output file.')
  cmd.add_argument('--model', required=True, help='the path to the model file.')
  args = cmd.parse_args()

  train_sents = load_corpus(args.train)
  x_train = [sent2features(s) for s in train_sents]
  y_train = [sent2labels(s) for s in train_sents]
  print('# training data: {0}'.format(len(x_train)))

  devel_sents = load_corpus(args.devel)
  x_test = [sent2features(s) for s in devel_sents]
  y_test = [sent2labels(s) for s in devel_sents]
  print('# test data: {0}'.format(len(x_test)))

  trainer = pycrfsuite.Trainer(algorithm='lbfgs', verbose=True)

  for xseq, yseq in zip(x_train, y_train):
    trainer.append(xseq, yseq)

  trainer.set_params({
    'c1': 0.0,  # coefficient for L1 penalty
    'c2': 1e-5,  # coefficient for L2 penalty
    'max_iterations': 1000,  # stop earlier
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
  })

  print(trainer.params())

  trainer.train(args.model)

  tagger = pycrfsuite.Tagger()
  tagger.open(args.model)

  if args.output is not None:
    fpo = codecs.open(args.output, 'w', encoding='utf-8')
  else:
    fpo = sys.stdout
  pred = []
  for xseq, yseq in zip(x_test, y_test):
    output = tagger.tag(sent2features(xseq))
    print(output, file=fpo)
    pred.append(output)
  print(f_score(y_test, pred), file=sys.stderr)


if __name__ == "__main__":
  main()
