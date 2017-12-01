#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.nn.utils.rnn as rnn
from compiler.ast import flatten

#import cuda_functional as MF
import dataloader
import modules

import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

torch.manual_seed(31415926)
random.seed(31415926)
tag_to_ix = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
ix_to_tag = {ix:tag for tag, ix in tag_to_ix.items()}

class Model(nn.Module):
  def __init__(self, args, uni_emb_layer, bi_emb_layer, n_class = 4):
    super(Model, self).__init__()
    self.args = args
    self.uni_emb_layer = uni_emb_layer
    self.bi_emb_layer = bi_emb_layer
    if args.cnn:
      self.encoder = modules.MultiLayerCNN(
          uni_emb_layer.n_d + bi_emb_layer.n_d * 2,
          args.hidden_dim,
          args.depth,
          args.dropout
          )
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim, n_class)
    elif args.gcnn:
      self.encoder = modules.GatedCNN(
          uni_emb_layer.n_d + bi_emb_layer.n_d * 2,
          args.hidden_dim,
          args.depth,
          args.dropout
        )
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim, n_class)
    elif args.lstm:
      self.encoder = nn.LSTM(
        uni_emb_layer.n_d + bi_emb_layer.n_d * 2, 
        args.hidden_dim, 
        num_layers = args.depth, 
        bidirectional = True, 
        batch_first = False,
        dropout = args.dropout)
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim * 2, n_class)      
    elif args.dilated:
      self.encoder = modules.DilatedCNN(
          uni_emb_layer.n_d + bi_emb_layer.n_d * 2,
          args.hidden_dim,
          args.depth,
          args.dropout
        )
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim, n_class)
    elif args.sru:
      self.encoder = MF.SRU(
        uni_emb_layer.n_d + bi_emb_layer.n_d * 2,
        args.hidden_dim,
        args.depth,
        dropout = args.dropout,
        use_tanh = 1,
        bidirectional = True
      )
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim * 2, n_class)
    self.train_time = 0
    self.eval_time = 0
    self.emb_time = 0
    self.classify_time = 0

  def forward(self, x, y):
    
    start_time = time.time()    
    unigram = Variable(x[0])

    lb_indices = Variable(torch.LongTensor(range(x[1].size(1))[:-1]))
    rb_indices = Variable(torch.LongTensor(range(x[1].size(1))[1:]))
    

    left_bigram = torch.index_select(Variable(x[1]), 1, lb_indices)  # expect the end
    right_bigram = torch.index_select(Variable(x[1]), 1, rb_indices)  # expect the start

    unigram = self.uni_emb_layer(unigram)
    left_bigram = self.bi_emb_layer(left_bigram)
    right_bigram = self.bi_emb_layer(right_bigram)

    emb = torch.cat((unigram, left_bigram, right_bigram), 2)  # cat those feature as the final features

    emb = F.dropout(emb, self.args.dropout, self.training)

    if not self.training:
      self.emb_time += time.time() - start_time

    start_time = time.time()
    if self.args.lstm or self.args.sru:
      x = emb.permute(1, 0, 2)
      #x = rnn.pack_padded_sequence(x, [len(seq) for seq in y])
      output, hidden = self.encoder(x)
      #output = rnn.pad_packed_sequence(output)[0]
      output = output.permute(1, 0, 2)  # the permuate is used to rearange those row
      
    elif self.args.cnn or self.args.gcnn:
      output = self.encoder(emb)
    elif self.args.dilated:
      output = self.encoder(emb)

    if self.training:
      self.train_time += time.time() - start_time
    else:
      self.eval_time += time.time() - start_time

    start_time = time.time()

    output, loss = self.classify_layer.forward(output, y)  # through a classify layer

    if not self.training:
      self.classify_time += time.time() - start_time

    return output, loss
  def model_save(self, model, path):
      """
      保存模型
      :param model:
      :return:
      """
      # torch.save(model.state_dict(), '../data/model.pkl')

      torch.save(model.state_dict(), path)

def get_intervals(tag):
  intervals = []
  l = len(tag)
  i = 0
  while (i < l):
    if (tag[i] == 2 or tag[i] == 3):
      intervals.append((i, i))
      i += 1
      continue
    j = i + 1
    while (True):
      if (j == l or tag[j] == 0 or tag[j] == 3):
        intervals.append((i, j - 1))
        i = j
        break
      elif (tag[j] == 2):
        intervals.append((i, j))
        i = j + 1
        break
      else:
        j += 1
  return intervals

def evaluate(gold, predicted):
  assert len(gold) == len(predicted)
  tp = 0
  fp = 0
  fn = 0
  for i in range(len(gold)):
    gold_intervals = get_intervals(gold[i])
    predicted_intervals = get_intervals(predicted[i])
    seg = set()
    for interval in gold_intervals:
      seg.add(interval)
      fn += 1
    for interval in predicted_intervals:
      if (interval in seg):
        tp += 1
        fn -= 1
      else:
        fp += 1
  P = 0 if tp == 0 else 1.0 * tp / (tp + fp)
  R = 0 if tp == 0 else 1.0 * tp / (tp + fn)
  F = 0 if P * R == 0 else 2.0 * P * R / (P + R)
  return P, R, F

def eval_model(niter, model, valid_x, valid_y):
  start_time = time.time()
  model.eval()
  N = len(valid_x)
  #total_loss = 0.0
  pred = []
  gold = []
  for x, y in zip(valid_x, valid_y):
    output, loss = model.forward(x, y)
    # print(output)
    # output batch*max_len

    # for index, value in enumerate(output):
    #     for index_tag, value_tag in enumerate(value):
    #         print ix_to_tag[value_tag],
    #     print('\n')
    #total_loss += loss.data[0]
    pred += output

    #print(pred)
    gold += y
  model.train()
  correct = map(cmp, flatten(gold), flatten(pred)).count(0)
  total = len(flatten(gold))
  P, R, F = evaluate(gold, pred)
  logging.info("**Evaluate result: acc = {:.6f}, P = {:.6f}, R = {:.6f}, F = {:.6f} time = {}".format(
    1.0 * correct / total, P, R, F, time.time() - start_time
  ))
  return F

def train_model(args, epoch, model, optimizer,
        train_x, train_y, valid_x, valid_y,
        test_x, test_y,
        best_valid, test_result, uni_emb_layer):

  model.train()
  args = model.args
  N = len(train_x[0])  # get the number of training_data
  niter = epoch*len(train_x[0])  # for statis and output log


  total_loss = 0.0
  total_tag = 0
  cnt = 0
  start_time = time.time()
  for x, y in zip(train_x, train_y):
    niter += 1
    cnt += 1
    model.zero_grad()
    output, loss = model.forward(x, y) # x,y batch*sentence_len
    total_loss += loss.data[0]
    total_tag += len(flatten(output))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
    optimizer.step()
    if (cnt * args.batch_size % 1024 == 0):
      logging.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={}".format(
        epoch, cnt,
        optimizer.param_groups[0]['lr'],
        1.0 * total_loss / total_tag,
        time.time() - start_time
      ))
      start_time = time.time()
    
  valid_result = eval_model(niter, model, valid_x, valid_y)

  logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_F1={:.6f}".format(
    epoch, niter,
    optimizer.param_groups[0]['lr'],
    loss.data[0],
    valid_result
  ))

  if valid_result > best_valid:
    model.model_save(model, args.model_save_path)
    best_valid = valid_result
    test_result = eval_model(niter, model, test_x, test_y)
    logging.info("Epoch={} iter={} lr={:.6f} test_F1={:.6f}".format(
      epoch, niter,
      optimizer.param_groups[0]['lr'],
      test_result
    ))

  sys.stdout.write("\n")
  return best_valid, test_result

def get_batch_res(x, predict_y, uni_emb_layer):
    """
    得到一个batch下的x, predcit_y， 特别注意它的x是pad的，但是y不是pad的
    :param x:
    :param predict_y:
    :param bi_emb_layer  用来映射word2id
    :return:   type:str
    """
    str_res = ""
    batch_size = len(x)  # 这个batch_size有多大
    for i in range(batch_size):
        input = []
        predict = []
        sentence_x = x[i][0].numpy().tolist()
        sentence_y = predict_y[i]
        for index, value in enumerate(sentence_y):  # 必须以y的长度为主，因为x是pad过的
            input.append(uni_emb_layer.id2word[sentence_x[index]])
            #print(sentence_y[index])
            predict.append(ix_to_tag[sentence_y[index]])
        str_res +=' '.join(input)
        str_res += '\t'
        str_res +=' '.join(predict)
        str_res += '\n'

    return str_res

def predict_model(model, uni_emb_layer, model_path, test_x, test_y, res_path):
    """
    predict the test data and output the segged word
    :param test_x:   gold input
    :param test_y:   gold laber
    :param res_path  the path for saving result
    :param uni_emb_layer
    :param model_path  the path of model
    :return:
    """

    fp_res = open(res_path, 'w')

    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    model.eval()
    N = len(test_x)
    str_res = ""
    pred = []
    gold = []
    for x, y in zip(test_x, test_y):
        output, loss = model.forward(x, y)

        # total_loss += loss.data[0]
        pred += output

        str_res += get_batch_res(x, output, uni_emb_layer)
        # print(pred)
        gold += y
    model.train()
    correct = map(cmp, flatten(gold), flatten(pred)).count(0)
    total = len(flatten(gold))
    P, R, F = evaluate(gold, pred)
    logging.info("**Evaluate result: acc = {:.6f}, P = {:.6f}, R = {:.6f}, F = {:.6f} time = {}".format(
        1.0 * correct / total, P, R, F, time.time() - start_time
    ))
    fp_res.write(str_res)
    fp_res.close()
    return F


def main(args):
  train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_data(args.path)
  uni_data = train_x[0] + valid_x[0] + test_x[0]
  bi_data = train_x[1] + valid_x[1] + test_x[1]

  logging.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
    len(train_y), len(valid_y), len(test_y)))

  logging.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
    sum([len(seq) for seq in train_y]), sum([len(seq) for seq in valid_y]), sum([len(seq) for seq in test_y])))


  uni_emb_layer = modules.EmbeddingLayer(
    args.d, uni_data, fix_emb = False,
    embs = dataloader.load_embedding(args.unigram_embedding))
    
  logging.info('unigram embedding size: ' + str(len(uni_emb_layer.word2id)))
  logging.info('unigram embedding size: ' + str(len(uni_emb_layer.id2word)))
  bi_emb_layer = modules.EmbeddingLayer(
    args.d, bi_data, fix_emb = False,
    embs = dataloader.load_embedding(args.bigram_embedding))

  logging.info('bigram embedding size: ' + str(len(bi_emb_layer.word2id)))

  nclasses = 4

  train_x, train_y = dataloader.create_batches(
    train_x, train_y,
    args.batch_size,
    uni_emb_layer.word2id, 
    bi_emb_layer.word2id, 
  )
  valid_x, valid_y = dataloader.create_batches(
    valid_x, valid_y,
    args.batch_size,
    uni_emb_layer.word2id, 
    bi_emb_layer.word2id, 
  )
  test_x, test_y = dataloader.create_batches(
    test_x, test_y,
    args.batch_size,
    uni_emb_layer.word2id, 
    bi_emb_layer.word2id, 
  )

  model = Model(args, uni_emb_layer, bi_emb_layer, nclasses)

  if args.type == 'test':
      predict_model(model, uni_emb_layer, args.model_save_path, test_x, test_y, args.res_path)
  else:  # if train, the procedure is the same as origin code
      need_grad = lambda x: x.requires_grad
      if args.adam:
        optimizer = optim.Adam(
          model.parameters(),
          filter(need_grad, model.parameters()),
          lr = args.lr
        )
      else:
        optimizer = optim.SGD(
          filter(need_grad, model.parameters()),
          lr = args.lr
        )
      best_valid = -1e+8
      test_result = -1e+8
      for epoch in range(args.max_epoch):
        best_valid, test_result = train_model(args, epoch, model, optimizer,
          train_x, train_y,
          valid_x, valid_y,
          test_x, test_y,
          best_valid, test_result,
          uni_emb_layer
        )
        if args.lr_decay>0:
          optimizer.param_groups[0]['lr'] *= args.lr_decay
        logging.info('Total encoder time: ' + str(model.eval_time))
        logging.info('Total embedding time: ' + str(model.emb_time))
        logging.info('Total classify time: ' + str(model.classify_time))

      logging.info("best_valid: {:.6f}".format(
        best_valid
      ))
      logging.info("test_err: {:.6f}".format(
        test_result
      ))


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
  argparser.add_argument("--type", help = "whether learn or test", default='learn')
  argparser.add_argument("--res_path", help = "test seg res path", default='../data/test_res.txt')
  argparser.add_argument("--gcnn", action='store_true', help="whether to use gated cnn")
  argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
  argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
  argparser.add_argument("--dilated", action='store_true', help="whether to use dialted CNN")
  argparser.add_argument("--sru", action='store_true', help="whether to use SRU")
  argparser.add_argument("--adam", action='store_true', help="whether to use adam")
  argparser.add_argument("--model_save_path", type=str, required=True, help="path to save model", default='../data/model.pkl')
  argparser.add_argument("--path", type=str, required=True, help="path to corpus directory", default='../data')
  argparser.add_argument("--unigram_embedding", type=str, required=True, help="unigram word vectors", default='../data/unigram_100.embed')
  argparser.add_argument("--bigram_embedding", type=str, required=True, help="bigram word vectors", default='../data/bigram_100.embed')
  argparser.add_argument("--batch_size", "--batch", type=int, default=32)
  argparser.add_argument("--hidden_dim", "--hidden", type=int, default=128)
  argparser.add_argument("--max_epoch", type=int, default=100)
  argparser.add_argument("--d", type=int, default=100)
  argparser.add_argument("--dropout", type=float, default=0.0)
  argparser.add_argument("--depth", type=int, default=2)
  argparser.add_argument("--lr", type=float, default=0.01)
  argparser.add_argument("--lr_decay", type=float, default=0)
  argparser.add_argument("--clip_grad", type=float, default=5)

  args = argparser.parse_args()
  print (args)
  main(args)

