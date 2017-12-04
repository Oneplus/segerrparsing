#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import sys
import argparse
import codecs
import pycrfsuite
from seqlabel.utils import flatten


# copied from CRFsuite example
def word2features(sent, i):
  word = sent[i][0]
  prev_word = sent[i - 1][0] if i > 0 else u'<BOS>'
  next_word = sent[i + 1][0] if i < len(sent) - 1 else u'<EOS>'
  prev_word2 = sent[i - 2][0] if i > 1 else u'<BOS>'
  next_word2 = sent[i + 2][0] if i < len(sent) - 2 else u'<EOS>'
  features = [
    u'w0={0}'.format(word),
    u'w-1={0}'.format(prev_word),
    u'w-2={0}'.format(prev_word2),
    u'w+1={0}'.format(next_word),
    u'w+2={0}'.format(next_word2),
    u'w-2={0}|w-1={1}'.format(prev_word2, prev_word),
    u'w-1={0}|w0={1}'.format(prev_word, word),
    u'w0={0}|w+1={1}'.format(word, next_word),
    u'w+1={0}|w+2={1}'.format(next_word, next_word2),
    u'w-2={0}|w-1={1}|w0={2}'.format(prev_word2, prev_word, word),
    u'w-1={0}|w0={1}|w+1={2}'.format(prev_word, word, next_word),
    u'w0={0}|w+1={1}|w+2={2}'.format(word, next_word, next_word2),
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
    tokens = line.strip().split()
    words = [token.rsplit('_', 1)[0] for token in tokens]
    tags = [token.rsplit('_', 1)[1] for token in tokens]
    sents.append(zip(words, tags))
  return sents


def main():
  cmd = argparse.ArgumentParser('A feature CRF baseline for word segmentation.')
  cmd.add_argument('--do_train', action='store_true', default=False, help='do training.')
  cmd.add_argument('--train', required=True, help='the path to the training file.')
  cmd.add_argument('--devel', required=True, help='the path to the development file.')
  cmd.add_argument('--output', help='the path to the output file.')
  cmd.add_argument('--model', required=True, help='the path to the model file.')
  args = cmd.parse_args()

  if args.do_train:
    train_sents = load_corpus(args.train)
    x_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    print('# training data: {0}'.format(len(x_train)))

  devel_sents = load_corpus(args.devel)
  x_test = [sent2features(s) for s in devel_sents]
  y_test = [sent2labels(s) for s in devel_sents]
  print('# test data: {0}'.format(len(x_test)))

  if args.do_train:
    trainer = pycrfsuite.Trainer(algorithm='lbfgs', verbose=True)

    for xseq, yseq in zip(x_train, y_train):
      trainer.append(xseq, yseq)

    trainer.set_params({
      'c1': 0.0,  # coefficient for L1 penalty
      'c2': 1e-5,  # coefficient for L2 penalty
      'max_iterations': 500,  # stop earlier
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
  gold, pred = [], []
  for i in range(len(devel_sents)):
    output = tagger.tag(sent2features(devel_sents[i]))
    print(' '.join(output), file=fpo)
    pred.append(output)
    gold.append([tag for ch, tag in devel_sents[i]])
  p = sum(map(lambda x, y: 1 if x == y else 0, flatten(pred), flatten(gold))) / len(flatten(gold))
  print(p, file=sys.stderr)


if __name__ == "__main__":
  main()
