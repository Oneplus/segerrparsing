#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import os
import codecs
import errno
import argparse
import time
import random
import torch
import logging
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from seqlabel.modules import MultiLayerCNN, GatedCNN, DilatedCNN, ClassifyLayer, EmbeddingLayer
from seqlabel.dataloader import load_embedding, pad
from seqlabel.utils import flatten, deep_iter, dict2namedtuple, f_score
try:
  import seqlabel.cuda_functional as MF
except:
  print('SRU is not supported.', file=sys.stderr)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)] %(message)s')
tag_to_ix = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}


def read_corpus(path):
  """
  read CoNLL format data.

  :param path:
  :return:
  """
  unigram_dataset, bigram_dataset, labels_dataset = [], [], []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for data in fin.read().strip().split('\n\n'):
      unigram, bigram, labels = [], [], []
      lines = data.splitlines()
      for line in lines:
        tokens = line.split()
        if len(tokens) == 6:
          tokens.insert(1, '\u3000')
          tokens.insert(2, '\u3000')
        chars = tokens[1]
        unigram.extend(list(chars))
        if len(chars) == 1:
          labels.append(3)
        else:
          for j in range(len(chars)):
            if j == 0:
              labels.append(0)
            elif j == len(chars) - 1:
              labels.append(2)
            else:
              labels.append(1)
      bigram.append('<s>' + unigram[0])
      for i in range(len(unigram) - 1):
        bigram.append(unigram[i] + unigram[i + 1])
      bigram.append(unigram[-1] + '</s>')

      unigram_dataset.append(unigram)
      bigram_dataset.append(bigram)
      labels_dataset.append(labels)
  return (unigram_dataset, bigram_dataset), labels_dataset


def read_data(train_path, valid_path, test_path):
  train_x, train_y = read_corpus(train_path)
  valid_x, valid_y = read_corpus(valid_path)
  test_x, test_y = read_corpus(test_path)
  return train_x, train_y, valid_x, valid_y, test_x, test_y


def create_one_batch(x, y, uni_map2id, bi_map2id, oov='<oov>', sort=True, use_cuda=False):
  """

  :param x: list[(list(str), list(str))]
  :param y: list[list[int]]
  :param batch_size: int
  :param uni_map2id: dict[str, int]
  :param bi_map2id: dict[str, int]
  :param oov:
  :param sort:
  :param use_cuda:
  :return:
  """
  lst = list(range(len(x[0])))
  if sort:
    lst.sort(key=lambda i_: -len(x[0][i_]))  # descent sort

  x1 = [x[0][i] for i in lst]
  x2 = [x[1][i] for i in lst]
  y = [y[i] for i in lst]

  oov_id = uni_map2id[oov]
  uni = pad(x1)  # now, uni is the result after padding
  uni_length = len(uni[0])
  batch_size = len(uni)
  uni = [uni_map2id.get(w, oov_id) for seq in uni for w in seq]  # convert to single list
  uni = torch.LongTensor(uni)

  assert uni.size(0) == uni_length * batch_size

  oov_id = bi_map2id[oov]
  bi = pad(x2)  # bi is the result after padding
  bi_length = len(bi[0])
  bi = [bi_map2id.get(w, oov_id) for seq in bi for w in seq]  # represented by id
  bi = torch.LongTensor(bi)

  assert bi.size(0) == bi_length * batch_size

  x1, x2 = (uni.view(batch_size, uni_length).contiguous(), bi.view(batch_size, bi_length).contiguous())
  if use_cuda:
    x1 = x1.cuda()
    x2 = x2.cuda()

  return (x1, x2), y


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, uni_map2id, bi_map2id, perm=None,
                   shuffle=True, sort=True, use_cuda=False, text=None):
  """

  :param x: list[(list(str), list(str))]
  :param y: list[list[int]]
  :param batch_size: int
  :param uni_map2id: dict[str, int]
  :param bi_map2id: dict[str, int]
  :param perm:
  :param shuffle:
  :param sort:
  :param use_cuda:
  :param text:
  :return:
  """
  lst = perm or list(range(len(x[0])))
  if shuffle:
    random.shuffle(lst)

  # sort sequences based on their length; necessary for SST
  if sort:
    lst.sort(key=lambda i_: -len(x[0][i_]))

  x = ([x[0][i] for i in lst], [x[1][i] for i in lst])  # descend by len(sentence)
  y = [y[i] for i in lst]
  if text is not None:
    text = [text[0][i] for i in lst]

  sum_len = 0.0
  batches_x = []
  batches_y = []
  batches_text = []
  size = batch_size
  nbatch = (len(x[0]) - 1) // size + 1  # the number of batch
  ninst = 0
  for i in range(nbatch):
    start_id, end_id = i * size, (i + 1) * size
    bx, by = create_one_batch((x[0][start_id: end_id], x[1][start_id: end_id]), y[start_id: end_id],
                              uni_map2id, bi_map2id, sort=sort, use_cuda=use_cuda)
    sum_len += sum([len(y_) for y_ in by])
    ninst += len(by)
    batches_x.append(bx)
    batches_y.append(by)
    if text is not None:
      batches_text.append(text[start_id: end_id])

  if sort:
    perm = list(range(nbatch))
    random.shuffle(perm)
    batches_x = [batches_x[i] for i in perm]
    batches_y = [batches_y[i] for i in perm]
    if text is not None:
      batches_text = [batches_text[i] for i in perm]

  logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / ninst))

  if text is not None:
    return batches_x, batches_y, batches_text
  return batches_x, batches_y


class Model(nn.Module):
  def __init__(self, args, uni_emb_layer, bi_emb_layer, n_class=4, use_cuda=False):
    super(Model, self).__init__()
    self.args = args
    self.use_cuda = use_cuda
    self.uni_emb_layer = uni_emb_layer
    self.bi_emb_layer = bi_emb_layer

    input_dim = uni_emb_layer.n_d + bi_emb_layer.n_d * 2
    if args.encoder.lower() == 'cnn':
      self.encoder = MultiLayerCNN(input_dim, args.hidden_dim, args.depth, args.dropout)
      encoded_dim = args.hidden_dim
    elif args.encoder.lower() == 'gcnn':
      self.encoder = GatedCNN(input_dim, args.hidden_dim, args.depth, args.dropout)
      encoded_dim = args.hidden_dim
    elif args.encoder.lower() == 'lstm':
      self.encoder = nn.LSTM(input_dim, args.hidden_dim, num_layers=args.depth, bidirectional=True,
                             batch_first=False, dropout=args.dropout)
      encoded_dim = args.hidden_dim * 2
    elif args.encoder.lower() == 'dilated':
      self.encoder = DilatedCNN(input_dim, args.hidden_dim, args.depth, args.dropout)
      encoded_dim = args.hidden_dim
    elif args.encoder.lower() == 'sru':
      self.encoder = MF.SRU(input_dim, args.hidden_dim, args.depth, dropout=args.dropout, use_tanh=1,
                            bidirectional=True)
      encoded_dim = args.hidden_dim * 2
    else:
      raise ValueError('Unknown encoder: {0}'.format(args.encoder))

    self.classify_layer = ClassifyLayer(encoded_dim, n_class, use_cuda=use_cuda)
    self.train_time = 0
    self.eval_time = 0
    self.emb_time = 0
    self.classify_time = 0

  def forward(self, x, y):
    start_time = time.time()

    unigram = Variable(x[0]).cuda() if self.use_cuda else Variable(x[0])

    lb_indices = Variable(torch.LongTensor(range(x[1].size(1))[:-1]))
    rb_indices = Variable(torch.LongTensor(range(x[1].size(1))[1:]))
    if self.use_cuda:
      lb_indices = lb_indices.cuda()
      rb_indices = rb_indices.cuda()

    left_bigram = torch.index_select(Variable(x[1]).cuda() if self.use_cuda else Variable(x[1]), 1, lb_indices)
    right_bigram = torch.index_select(Variable(x[1]).cuda() if self.use_cuda else Variable(x[1]), 1, rb_indices)

    unigram = self.uni_emb_layer(unigram)
    left_bigram = self.bi_emb_layer(left_bigram)
    right_bigram = self.bi_emb_layer(right_bigram)

    emb = torch.cat((unigram, left_bigram, right_bigram), 2)  # cat those feature as the final features
    emb = F.dropout(emb, self.args.dropout, self.training)

    if not self.training:
      self.emb_time += time.time() - start_time

    start_time = time.time()
    if self.args.encoder.lower() in ('lstm', 'sru'):
      x = emb.permute(1, 0, 2)
      output, hidden = self.encoder(x)
      output = output.permute(1, 0, 2)  # the permuate is used to rearange those row
    elif self.args.encoder.lower() in ('cnn', 'gcnn'):
      output = self.encoder(emb)
    elif self.args.encoder.lower() == 'dilated':
      output = self.encoder(emb)
    else:
      raise ValueError('Unknown encoder: {0}'.format(self.encoder))

    if self.training:
      self.train_time += time.time() - start_time
    else:
      self.eval_time += time.time() - start_time

    start_time = time.time()
    output, loss = self.classify_layer.forward(output, y)  # through a classify layer

    if not self.training:
      self.classify_time += time.time() - start_time

    return output, loss


def eval_model(model, valid_x, valid_y):
  start_time = time.time()
  model.eval()
  # total_loss = 0.0
  pred, gold = [], []
  for x, y in zip(valid_x, valid_y):
    output, loss = model.forward(x, y)
    pred += output
    gold += y

  model.train()
  correct = map(cmp, flatten(gold), flatten(pred)).count(0)
  total = len(flatten(gold))
  p, r, f = f_score(gold, pred)
  logging.info("**Evaluate result: acc={:.6f}, P={:.6f}, R={:.6f}, F={:.6f} time={:.2f}s.".format(
    1.0 * correct / total, p, r, f, time.time() - start_time))
  return f


def train_model(epoch, model, optimizer,
                train_x, train_y, valid_x, valid_y, test_x, test_y,
                best_valid, test_result):
  model.train()
  args = model.args
  niter = epoch * len(train_x[0])  # for statis and output log

  total_loss = 0.0
  total_tag = 0
  cnt = 0
  start_time = time.time()

  # shuffle the data
  lst = list(range(len(train_x)))
  random.shuffle(lst)
  train_x, train_y = [train_x[l] for l in lst], [train_y[l] for l in lst]

  for x, y in zip(train_x, train_y):
    niter += 1
    cnt += 1
    model.zero_grad()
    output, loss = model.forward(x, y)  # x,y batch*sentence_len
    total_loss += loss.data[0]
    total_tag += len(flatten(output))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
    optimizer.step()
    if cnt * args.batch_size % 1024 == 0:
      logging.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={:.2f}s.".format(
        epoch, cnt, optimizer.param_groups[0]['lr'], 1.0 * loss.data[0] / total_tag, time.time() - start_time))
      start_time = time.time()

  valid_result = eval_model(model, valid_x, valid_y)
  logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_F1={:.6f}".format(
    epoch, niter, optimizer.param_groups[0]['lr'], total_loss, valid_result))

  if valid_result > best_valid:
    torch.save(model.state_dict(), os.path.join(args.model, 'model.pkl'))
    best_valid = valid_result
    test_result = eval_model(model, test_x, test_y)
    logging.info('New best achieved.')
    logging.info("Epoch={} iter={} lr={:.6f} test_F1={:.6f}".format(
      epoch, niter, optimizer.param_groups[0]['lr'], test_result))

  return best_valid, test_result


def train():
  cmd = argparse.ArgumentParser('{0} train'.format(__file__))
  cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
  cmd.add_argument('--gpu', default=-1, type=int, help='The id of gpu, -1 if cpu.')
  cmd.add_argument('--encoder', default='lstm', choices=['lstm'],
                   help='the type of encoder: valid options=[lstm, sru, gcnn, cnn, dilated]')
  cmd.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'],
                   help='the type of optimizer: valid options=[sgd, adam]')
  cmd.add_argument('--train_path', required=True, help='the path to the training file.')
  cmd.add_argument('--valid_path', required=True, help='the path to the validation file.')
  cmd.add_argument('--test_path', required=True, help='the path to the testing file.')
  cmd.add_argument("--model", required=True, help="path to save model")
  cmd.add_argument("--unigram_embedding", type=str, required=True, help="unigram word vectors")
  cmd.add_argument("--bigram_embedding", type=str, required=True, help="bigram word vectors")
  cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
  cmd.add_argument("--hidden_dim", "--hidden", type=int, default=128, help='the hidden dimension.')
  cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
  cmd.add_argument("--d", type=int, default=100, help='the input dimension.')
  cmd.add_argument("--dropout", type=float, default=0.0, help='the dropout rate')
  cmd.add_argument("--depth", type=int, default=2, help='the depth of lstm')
  cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
  cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
  cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')
  opt = cmd.parse_args(sys.argv[2:])
  print(opt)

  torch.manual_seed(opt.seed)
  random.seed(opt.seed)
  if opt.gpu >= 0:
    torch.cuda.set_device(opt.gpu)
    if opt.seed > 0:
      torch.cuda.manual_seed(opt.seed)

  use_cuda = opt.gpu >= 0 and torch.cuda.is_available()
  train_x, train_y, valid_x, valid_y, test_x, test_y = read_data(opt.train_path, opt.valid_path, opt.test_path)

  logging.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
    len(train_y), len(valid_y), len(test_y)))

  logging.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
    sum([len(seq) for seq in train_y]), sum([len(seq) for seq in valid_y]), sum([len(seq) for seq in test_y])))

  def extend(words, word2id, oov='<oov>', pad='<pad>'):
    for w in deep_iter(words):
      if w not in word2id:
        word2id[w] = len(word2id)
    if oov not in word2id:
      word2id[oov] = len(word2id)
    if pad not in word2id:
      word2id[pad] = len(word2id)

  uni_embs_words, uni_embs = load_embedding(opt.unigram_embedding)
  uni_lexicon = {unigram: i for i, unigram in enumerate(uni_embs_words)}
  extend(train_x[0], uni_lexicon)
  uni_emb_layer = EmbeddingLayer(opt.d, uni_lexicon, fix_emb=False, embs=(uni_embs_words, uni_embs))
  logging.info('unigram embedding size: {0}'.format(len(uni_emb_layer.word2id)))

  bi_embs_words, bi_embs = load_embedding(opt.bigram_embedding)
  bi_lexicon = {bigram: i for i, bigram in enumerate(bi_embs_words)}
  extend(train_x[1], bi_lexicon)
  bi_emb_layer = EmbeddingLayer(opt.d, bi_lexicon, fix_emb=False, embs=(bi_embs_words, bi_embs))
  logging.info('bigram embedding size: {0}'.format(len(bi_emb_layer.word2id)))

  nclasses = 4
  train_x, train_y = create_batches(
    train_x, train_y, opt.batch_size, uni_emb_layer.word2id, bi_emb_layer.word2id,
    use_cuda=use_cuda)
  valid_x, valid_y = create_batches(
    valid_x, valid_y, opt.batch_size, uni_emb_layer.word2id, bi_emb_layer.word2id,
    shuffle=False, sort=False, use_cuda=use_cuda)
  test_x, test_y = create_batches(
    test_x, test_y, opt.batch_size, uni_emb_layer.word2id, bi_emb_layer.word2id,
    shuffle=False, sort=False, use_cuda=use_cuda)

  model = Model(opt, uni_emb_layer, bi_emb_layer, nclasses, use_cuda=use_cuda)
  if use_cuda:
    model = model.cuda()

  need_grad = lambda x: x.requires_grad
  if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=opt.lr)
  else:
    optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=opt.lr)

  try:
    os.makedirs(opt.model)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

  # save unigram dict.
  with codecs.open(os.path.join(opt.model, 'unigram.dic'), 'w', encoding='utf-8') as fpo:
    for word, i in uni_emb_layer.word2id.items():
      print('{0}\t{1}'.format(word, i), file=fpo)

  # save bigram dict.
  with codecs.open(os.path.join(opt.model, 'bigram.dic'), 'w', encoding='utf-8') as fpo:
    for word, i in bi_emb_layer.word2id.items():
      print('{0}\t{1}'.format(word, i), file=fpo)

  json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))
  best_valid, test_result = -1e8, -1e8
  for epoch in range(opt.max_epoch):
    best_valid, test_result = train_model(epoch, model, optimizer, train_x, train_y, valid_x, valid_y,
                                          test_x, test_y,
                                          best_valid, test_result)
    if opt.lr_decay > 0:
      optimizer.param_groups[0]['lr'] *= opt.lr_decay
    logging.info('Total encoder time: {:.2f}s.'.format(model.eval_time))
    logging.info('Total embedding time: {:.2f}s.'.format(model.emb_time))
    logging.info('Total classify time: {:.2f}s.'.format(model.classify_time))

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
  # load unigram
  uni_lexicon = {}
  with codecs.open(os.path.join(args.model, 'unigram.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      token, i = line.strip().split('\t')
      uni_lexicon[token] = int(i)
  uni_emb_layer = EmbeddingLayer(args2.d, uni_lexicon, fix_emb=False, embs=None)
  logging.info('unigram embedding size: ' + str(len(uni_emb_layer.word2id)))

  # load bigram
  bi_lexicon = {}
  with codecs.open(os.path.join(args.model, 'bigram.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      token, i = line.strip().split('\t')
      bi_lexicon[token] = int(i)
  bi_emb_layer = EmbeddingLayer(args2.d, bi_lexicon, fix_emb=False, embs=None)
  logging.info('bigram embedding size: ' + str(len(bi_emb_layer.word2id)))

  use_cuda = args.cuda and torch.cuda.is_available()
  model = Model(args2, uni_emb_layer, bi_emb_layer, use_cuda=use_cuda)

  if use_cuda:
    model = model.cuda()

  model.load_state_dict(torch.load(os.path.join(args.model, 'model.pkl')))

  test_x, test_y = read_corpus(args.input)
  test_x, test_y, test_text = create_batches(test_x, test_y, args2.batch_size, uni_lexicon, bi_lexicon,
                                             shuffle=False, sort=False, use_cuda=use_cuda, text=test_x)

  if args.output is not None:
    fpo = codecs.open(args.output, 'w', encoding='utf-8')
  else:
    fpo = codecs.getwriter('utf-8')(sys.stdout)
  start_time = time.time()
  model.eval()
  pred, gold = [], []
  for x, y, text in zip(test_x, test_y, test_text):
    output, loss = model.forward(x, y)
    pred += output
    gold += y
    batch_size = len(text)
    for bid in range(batch_size):
      words, word = [], ''
      for ch, tag in zip(text[bid], output[bid]):
        tag = ix_to_tag[tag]
        if tag in ('B', 'S'):
          if len(word) > 0:
            words.append(word)
          word = ch
        else:
          word += ch
      words.append(word)
      for k, word in enumerate(words):
        print('{0}\t{1}\t{1}\t_\t_\t_\t_\t_'.format(k + 1, word, word), file=fpo)
      print(file=fpo)

  correct = map(cmp, flatten(gold), flatten(pred)).count(0)
  total = len(flatten(gold))
  p, r, f = f_score(gold, pred)
  logging.info("**Evaluate result: acc={:.6f}, P={:.6f}, R={:.6f}, F={:.6f} time={:.2f}s.".format(
    1.0 * correct / total, p, r, f, time.time() - start_time))


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train()
  elif len(sys.argv) > 1 and sys.argv[1] == 'test':
    test()
  else:
    print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
