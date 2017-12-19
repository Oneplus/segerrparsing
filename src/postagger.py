#!/usr/bin/env python
from __future__ import print_function
import os
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from seqlabel.modules import MultiLayerCNN, GatedCNN, DilatedCNN, ClassifyLayer, EmbeddingLayer, PartialClassifyLayer
from seqlabel.dataloader import load_embedding, pad
from seqlabel.utils import flatten, deep_iter, dict2namedtuple
try:
  import seqlabel.cuda_functional as MF
except:
  print('SRU is not supported.', file=sys.stderr)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def read_corpus(path):
  data = []
  labels = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin:
      terms = line.strip().split()
      data.append([])
      labels.append([])
      for term in terms:
        data[-1].append(term.split(u'_')[0])
        labels[-1].append(term.split(u'_')[1])
  return data, labels


def read_data(train_path, valid_path, test_path):
  train_x, train_y = read_corpus(train_path)
  valid_x, valid_y = read_corpus(valid_path)
  test_x, test_y = read_corpus(test_path)
  return train_x, train_y, valid_x, valid_y, test_x, test_y


def create_one_batch(x, y, map2id, oov='<oov>', use_cuda=False):
  lst = range(len(x))
  lst = sorted(lst, key=lambda i: -len(y[i]))

  x = [x[i] for i in lst]
  y = [y[i] for i in lst]

  oov_id = map2id[oov]
  x = pad(x)
  length = len(x[0])
  batch_size = len(x)
  x = [map2id.get(w, oov_id) for seq in x for w in seq]
  x = torch.LongTensor(x)
  assert x.size(0) == length * batch_size
  x = x.view(batch_size, length).contiguous()
  if use_cuda:
    x = x.cuda()
  return x, y


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=True, use_cuda=False, text=None):
  lst = perm or range(len(x))
  random.shuffle(lst)

  if sort:
    lst = sorted(lst, key=lambda i: -len(y[i]))

  x = [x[i] for i in lst]
  y = [y[i] for i in lst]
  if text is not None:
    text = [text[i] for i in lst]

  sum_len = 0.0
  batches_x = []
  batches_y = []
  batches_text = []
  size = batch_size
  nbatch = (len(x) - 1) // size + 1
  for i in range(nbatch):
    start_id, end_id = i * size, (i + 1) * size
    bx, by = create_one_batch(x[start_id: end_id], y[start_id: end_id], map2id, use_cuda=use_cuda)
    sum_len += len(by[0])
    batches_x.append(bx)
    batches_y.append(by)
    if text is not None:
      batches_text.append(text[start_id: end_id])

  if sort:
    perm = range(nbatch)
    random.shuffle(perm)
    batches_x = [batches_x[i] for i in perm]
    batches_y = [batches_y[i] for i in perm]
    if text is not None:
      batches_text = [batches_text[i] for i in perm]

  logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / nbatch))
  if text is not None:
    return batches_x, batches_y, batches_text
  return batches_x, batches_y


class Model(nn.Module):
  def __init__(self, args, emb_layer, n_class, use_cuda):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.args = args
    self.emb_layer = emb_layer
    encoder_output = None
    if args.encoder.lower() == 'cnn':
      self.encoder = MultiLayerCNN(emb_layer.n_d, args.hidden_dim, args.depth, args.dropout)
      encoder_output = args.hidden_dim
    elif args.encoder.lower() == 'gcnn':
      self.encoder = GatedCNN(emb_layer.n_d, args.hidden_dim, args.depth, args.dropout)
      encoder_output = args.hidden_dim
    elif args.encoder.lower() == 'lstm':
      self.encoder = nn.LSTM(emb_layer.n_d, args.hidden_dim, num_layers=args.depth, bidirectional=True,
                             batch_first=False, dropout=args.dropout)
      encoder_output = args.hidden_dim * 2
    elif args.encoder.lower() == 'dilated':
      self.encoder = DilatedCNN(emb_layer.n_d, args.hidden_dim, args.depth, args.dropout)
      encoder_output = args.hidden_dim
    elif args.encoder.lower() == 'sru':
      self.encoder = MF.SRU(emb_layer.n_d, args.hidden_dim, args.depth, dropout=args.dropout, use_tanh=1,
                            bidirectional=True)
      encoder_output = args.hidden_dim * 2
    self.classify_layer = ClassifyLayer(encoder_output, n_class, self.use_cuda)

    self.classify_partial = PartialClassifyLayer(encoder_output, n_class, self.use_cuda)

    self.train_time = 0
    self.eval_time = 0
    self.emb_time = 0
    self.classify_time = 0

  def forward(self, x, y):
    start_time = time.time()
    emb = self.emb_layer(Variable(x).cuda() if self.use_cuda else Variable(x))
    emb = F.dropout(emb, self.args.dropout, self.training)

    if not self.training:
      self.emb_time += time.time() - start_time

    start_time = time.time()
    if self.args.encoder.lower() in ('lstm', 'sru'):
      x = emb.permute(1, 0, 2)
      output, hidden = self.encoder(x)
      output = output.permute(1, 0, 2)
    elif self.args.encoder.lower() in ('cnn', 'gcnn'):
      output = self.encoder(emb)
    elif self.args.encoder.lower() == 'dilated':
      output = self.encoder(emb)
    else:
      raise ValueError('unknown encoder: {0}'.format(self.args.encoder))

    if self.training:
      self.train_time += time.time() - start_time
    else:
      self.eval_time += time.time() - start_time

    start_time = time.time()

    if self.args.use_partial == 'True':
      output, loss = self.classify_partial.forward(output, y)

    else:
      output, loss = self.classify_layer.forward(output, y)

    if not self.training:
      self.classify_time += time.time() - start_time

    return output, loss 


def eval_model(niter, model, valid_x, valid_y):
  start_time = time.time()
  model.eval()
  # total_loss = 0.0
  pred, gold = [], []
  for x, y in zip(valid_x, valid_y):
    output, loss = model.forward(x, y)
    # total_loss += loss.data[0]
    pred += output
    gold += y
  model.train()
  correct = map(cmp, flatten(gold), flatten(pred)).count(0)
  total = len(flatten(gold))
  logging.info("**Evaluate result: acc = {:.6f}, time = {}".format(1.0 * correct / total, time.time() - start_time))
  return 1.0 * correct / total


def train_model(epoch, model, optimizer,
                train_x, train_y, valid_x, valid_y,
                test_x, test_y,
                best_valid, test_result):
  model.train()
  args = model.args
  niter = epoch * len(train_x[0])

  total_loss = 0.0
  total_tag = 0
  cnt = 0
  start_time = time.time()
  for x, y in zip(train_x, train_y):
    niter += 1
    cnt += 1
    model.zero_grad()
    output, loss = model.forward(x, y)
    total_loss += loss.data[0]
    total_tag += len(flatten(output))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
    optimizer.step()
    if cnt * args.batch_size % 1024 == 0:
      logging.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={}".format(
        epoch, cnt,
        optimizer.param_groups[0]['lr'],
        1.0 * total_loss / total_tag,
        time.time() - start_time
      ))
      start_time = time.time()
    
  valid_result = eval_model(niter, model, valid_x, valid_y)

  logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}".format(epoch, niter,
                                                                                      optimizer.param_groups[0]['lr'],
                                                                                      loss.data[0], valid_result))

  if valid_result > best_valid:
    torch.save(model.state_dict(), os.path.join(args.model, 'model.pkl'))
    best_valid = valid_result
    test_result = eval_model(niter, model, test_x, test_y)
    logging.info("Epoch={} iter={} lr={:.6f} test_acc={:.6f}".format(epoch, niter, optimizer.param_groups[0]['lr'],
                                                                     test_result))
  return best_valid, test_result


def label_to_index(y, label_to_ix):
  for i in range(len(y)):
    for j in range(len(y[i])):
      if y[i][j] not in label_to_ix:
        label_to_ix[y[i][j]] = len(label_to_ix)
      y[i][j] = label_to_ix[y[i][j]]


def train():
  cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
  cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
  cmd.add_argument('--cuda', action='store_true', default=False, help='use cuda')
  cmd.add_argument('--encoder', default='lstm', help='the type of encoder: '
                                                     'valid options=[lstm, sru, gcnn, cnn, dilated]')
  cmd.add_argument('--optimizer', default='sgd', help='the type of optimizer: valid options=[sgd, adam]')
  cmd.add_argument('--train_path', required=True, help='the path to the training file.')
  cmd.add_argument('--valid_path', required=True, help='the path to the validation file.')
  cmd.add_argument('--test_path', required=True, help='the path to the testing file.')
  cmd.add_argument("--model", required=True, help="path to save model")
  cmd.add_argument("--word_embedding", type=str, required=True, help="word vectors")
  cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
  cmd.add_argument("--hidden_dim", "--hidden", type=int, default=128, help='the hidden dimension.')
  cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
  cmd.add_argument("--d", type=int, default=100, help='the input dimension.')
  cmd.add_argument("--dropout", type=float, default=0.0, help='the dropout rate')
  cmd.add_argument("--depth", type=int, default=2, help='the depth of lstm')
  cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
  cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
  cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')
  cmd.add_argument("--use_partial", default=False, help = "whether use the partial data")

  args = cmd.parse_args(sys.argv[2:])
  print(args)
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  use_cuda = args.cuda and torch.cuda.is_available()
  train_x, train_y, valid_x, valid_y, test_x, test_y = read_data(args.train_path, args.valid_path, args.test_path)
  logging.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
    len(train_y), len(valid_y), len(test_y)))
  logging.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
    sum([len(seq) for seq in train_y]), sum([len(seq) for seq in valid_y]), sum([len(seq) for seq in test_y])))

  label_to_ix = {'CIXIN':0}
  label_to_index(train_y, label_to_ix)
  label_to_index(valid_y, label_to_ix)
  label_to_index(test_y, label_to_ix)
  logging.info('number of tags: {0}'.format(len(label_to_ix)))

  def extend(words, word2id, oov='<oov>', pad='<pad>'):
    for w in deep_iter(words):
      if w not in word2id:
        word2id[w] = len(word2id)
    if oov not in word2id:
      word2id[oov] = len(word2id)
    if pad not in word2id:
      word2id[pad] = len(word2id)

  embs_words, embs = load_embedding(args.word_embedding)
  lexicon = {word: i for i, word in enumerate(embs_words)}
  extend(train_x, lexicon)
  emb_layer = EmbeddingLayer(args.d, lexicon, fix_emb=False, embs=(embs_words, embs))
  logging.info('embedding size: ' + str(len(emb_layer.word2id)))

  nclasses = len(label_to_ix)
  train_x, train_y = create_batches(train_x, train_y, args.batch_size, emb_layer.word2id, use_cuda=args.cuda)
  valid_x, valid_y = create_batches(valid_x, valid_y, args.batch_size, emb_layer.word2id, use_cuda=args.cuda)
  test_x, test_y = create_batches(test_x, test_y, args.batch_size, emb_layer.word2id, use_cuda=args.cuda)

  model = Model(args, emb_layer, nclasses, use_cuda)
  if use_cuda:
    model = model.cuda()

  need_grad = lambda x: x.requires_grad

  if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)
  else:
    optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=args.lr)

  try:
    os.makedirs(args.model)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

  # save unigram dict.
  with codecs.open(os.path.join(args.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
    for word, i in emb_layer.word2id.items():
      print(u'{0}\t{1}'.format(word, i), file=fpo)

  with codecs.open(os.path.join(args.model, 'label.dic'), 'w', encoding='utf-8') as fpo:
    for label, i in label_to_ix.items():
      print(u'{0}\t{1}'.format(label, i), file=fpo)

  json.dump(vars(args), codecs.open(os.path.join(args.model, 'config.json'), 'w', encoding='utf-8'))
  best_valid, test_result = -1e8, -1e8
  for epoch in range(args.max_epoch):
    best_valid, test_result = train_model(epoch, model, optimizer, train_x, train_y,
                                          valid_x, valid_y,
                                          test_x, test_y,
                                          best_valid, test_result)
    if args.lr_decay > 0:
      optimizer.param_groups[0]['lr'] *= args.lr_decay
    logging.info('Total encoder time: ' + str(model.eval_time))
    logging.info('Total embedding time: ' + str(model.emb_time))
    logging.info('Total classify time: ' + str(model.classify_time))

  logging.info("best_valid: {:.6f}".format(best_valid))
  logging.info("test_err: {:.6f}".format(test_result))


def test():
  cmd = argparse.ArgumentParser('The testing components of')
  cmd.add_argument('--cuda', action='store_true', default=False, help='use cuda')
  cmd.add_argument("--input", help="the path to the test file.")
  cmd.add_argument('--output', help='the path to the output file.')
  cmd.add_argument("--model", required=True, help="path to save model")
  args = cmd.parse_args(sys.argv[2:])

  args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))
  lexicon = {}
  with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      token, i = line.strip().split('\t')
      lexicon[token] = int(i)
  emb_layer = EmbeddingLayer(args2.d, lexicon, fix_emb=False, embs=None)
  logging.info('word embedding size: ' + str(len(emb_layer.word2id)))

  label2id, id2label = {}, {}
  with codecs.open(os.path.join(args.model, 'label.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      token, i = line.strip().split('\t')
      label2id[token] = int(i)
      id2label[int(i)] = token
  logging.info('number of labels: {0}'.format(len(label2id)))

  use_cuda = args.cuda and torch.cuda.is_available()
  model = Model(args2, emb_layer, len(label2id), use_cuda)
  model.load_state_dict(torch.load(os.path.join(args.model, 'model.pkl')))

  test_x, test_y = read_corpus(args.input)
  label_to_index(test_y, label2id)

  test_x, test_y, test_text = create_batches(test_x, test_y, args2.batch_size, lexicon, use_cuda=use_cuda, text=test_x)

  if args.output is not None:
    fpo = codecs.open(args.output, 'w', encoding='utf-8')
  else:
    fpo = sys.stdout
  start_time = time.time()
  model.eval()
  pred, gold = [], []
  for x, y, text in zip(test_x, test_y, test_text):
    output, loss = model.forward(x, y)
    pred += output
    gold += y
    for word, raw in zip(text, output):
      print(u'{0}\t{1}'.format(u' '.join(word), u' '.join([id2label[t] for t in raw])), file=fpo)

  correct = map(cmp, flatten(gold), flatten(pred)).count(0)
  total = len(flatten(gold))
  logging.info("**Evaluate result: acc = {:.6f}, time = {}".format(1.0 * correct / total, time.time() - start_time))


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train()
  elif len(sys.argv) > 1 and sys.argv[1] == 'test':
    test()
  else:
    print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
