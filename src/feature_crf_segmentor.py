#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse
import codecs
import pycrfsuite
from seqlabel.utils import f_score
from seqlabel.chinese import chartype


# copied from CRFsuite example
def word2features(sent, i):
  ch = sent[i][0]
  cht = chartype(ch)
  prev_ch = sent[i - 1][0] if i > 0 else u'<BOS>'
  next_ch = sent[i + 1][0] if i < len(sent) - 1 else u'<EOS>'
  prev_cht = chartype(prev_ch)
  next_cht = chartype(next_ch)
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
    u'ct-1={0}'.format(prev_cht),
    u'ct={0}'.format(cht),
    u'ct+1={0}'.format(next_cht)
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
  cmd.add_argument('--do_train', action='store_true', default=False, help='do training.')
  cmd.add_argument('--train', help='the path to the training file.')
  cmd.add_argument('--devel', help='the path to the development file.')
  cmd.add_argument('--algorithm', default='lbfgs', help='the learning algorithm')
  cmd.add_argument('--l1', type=float, default=0., help='the l1 tense')
  cmd.add_argument('--l2', type=float, default=1e-3, help='the l2 tense')
  cmd.add_argument('--output', help='the path to the output file.')
  cmd.add_argument('--model', required=True, help='the path to the model file.')
  args = cmd.parse_args()

  if args.do_train:
    train_sents = load_corpus(args.train)
    x_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    print('# training data: {0}'.format(len(x_train)))

    trainer = pycrfsuite.Trainer(algorithm=args.algorithm, verbose=True)

    for xseq, yseq in zip(x_train, y_train):
      trainer.append(xseq, yseq)

    if args.algorithm == 'lbfgs':
      trainer.set_params({'c1': args.l1, 'c2': args.l2,
                          'feature.possible_states': True,
                          'feature.possible_transitions': True})
    else:
      trainer.set_params({'c2': args.l2,
                          'feature.possible_states': True,
                          'feature.possible_transitions': True})

    print(trainer.params())
    trainer.train(args.model)

  devel_sents = load_corpus(args.devel)
  x_test = [sent2features(s) for s in devel_sents]
  y_test = [sent2labels(s) for s in devel_sents]
  print('# test data: {0}'.format(len(x_test)))

  tagger = pycrfsuite.Tagger()
  tagger.open(args.model)

  if args.output is not None:
    fpo = codecs.open(args.output, 'w', encoding='utf-8')
  else:
    fpo = sys.stdout
  gold, pred = [], []
  for i in range(len(devel_sents)):
    output = tagger.tag(sent2features(devel_sents[i]))
    print(' '.join(output), file=fpo)
    pred.append(output)
    gold.append([tag for ch, tag in devel_sents[i]])
  print(f_score(gold, pred), file=sys.stderr)


if __name__ == "__main__":
  main()
