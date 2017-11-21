#!/usr/bin/env python
from __future__ import print_function
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
#from compile.ast import flatten

import cuda_functional as MF
import dataloader
import modules

torch.manual_seed(31415926)
random.seed(31415926)


class Model(nn.Module):
  def __init__(self, args, emb_layer, n_class):
    super(Model, self).__init__()
    self.args = args
    self.emb_layer = emb_layer
    if args.cnn:
      self.encoder = modules.MultiLayerCNN(
          emb_layer.n_d,
          args.hidden_dim,
          args.depth,
          args.dropout
          )
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim, n_class)
    elif args.gcnn:
      self.encoder = modules.GatedCNN(
          emb_layer.n_d,
          args.hidden_dim,
          args.depth,
          args.dropout
          )
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim, n_class)
    elif args.lstm:
      self.encoder = nn.LSTM(
        emb_layer.n_d, 
        args.hidden_dim, 
        num_layers = args.depth, 
        bidirectional = True, 
        batch_first = False,
        dropout = args.dropout)
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim * 2, n_class)      
    elif args.dilated:
      self.encoder = modules.DilatedCNN(
          emb_layer.n_d,
          args.hidden_dim,
          args.depth,
          args.dropout
        )
      self.classify_layer = modules.ClassifyLayer(args.hidden_dim, n_class)
    elif args.sru:
      self.encoder = MF.SRU(
        emb_layer.n_d,
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
    emb = self.emb_layer(Variable(x).cuda())

    emb = F.dropout(emb, self.args.dropout, self.training)

    if not self.training:
      self.emb_time += time.time() - start_time

    start_time = time.time()
    if self.args.lstm or self.args.sru:
      x = emb.permute(1, 0, 2)
      output, hidden = self.encoder(x)
      output = output.permute(1, 0, 2)
    elif self.args.cnn or self.args.gcnn:
      output = self.encoder(emb)
    elif self.args.dilated:
      output = self.encoder(emb)

    if self.training:
      self.train_time += time.time() - start_time
    else:
      self.eval_time += time.time() - start_time

    start_time = time.time()

    output, loss = self.classify_layer.forward(output, y)

    if not self.training:
      self.classify_time += time.time() - start_time

    return output, loss 

def eval_model(niter, model, valid_x, valid_y):
  start_time = time.time()
  model.eval()
  N = len(valid_x)
  #total_loss = 0.0
  pred = []
  gold = []
  for x, y in zip(valid_x, valid_y):
    output, loss = model.forward(x, y)
    #total_loss += loss.data[0]
    pred += output
    gold += y
  model.train()
  correct = map(cmp, flatten(gold), flatten(pred)).count(0)
  total = len(flatten(gold))
  sys.stdout.write("**Evaluate result: acc = {:.6f}, time = {}\n".format(
            1.0 * correct / total, time.time() - start_time
          ))
  return 1.0 * correct / total

def train_model(epoch, model, optimizer,
        train_x, train_y, valid_x, valid_y,
        test_x, test_y,
        best_valid, test_result):

  model.train()
  args = model.args
  N = len(train_x[0])
  niter = epoch*len(train_x[0])

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
    if (cnt * args.batch_size % 1024 == 0):
      sys.stdout.write("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={}\n".format(
        epoch, cnt,
        optimizer.param_groups[0]['lr'],
        1.0 * total_loss / total_tag,
        time.time() - start_time
      ))
      start_time = time.time()
    
  valid_result = eval_model(niter, model, valid_x, valid_y)

  sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
    epoch, niter,
    optimizer.param_groups[0]['lr'],
    loss.data[0],
    valid_result
  ))

  if valid_result > best_valid:
    best_valid = valid_result
    test_result = eval_model(niter, model, test_x, test_y)
    sys.stdout.write("Epoch={} iter={} lr={:.6f} test_acc={:.6f}\n".format(
      epoch, niter,
      optimizer.param_groups[0]['lr'],
      test_result
    ))
  sys.stdout.write("\n")
  return best_valid, test_result


def label_to_index(y, label_to_ix):
  for i in range(len(y)):
    for j in range(len(y[i])):
      if y[i][j] not in label_to_ix:
        label_to_ix[y[i][j]] = len(label_to_ix)
      y[i][j] = label_to_ix[y[i][j]]


def main(args):
  train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_data(args.path)
  data = train_x + valid_x + test_x

  label_to_ix = {}

  label_to_index(train_y, label_to_ix)
  label_to_index(valid_y, label_to_ix)
  label_to_index(test_y, label_to_ix)

  print(len(train_x), len(train_y), len(valid_x), len(valid_y), len(test_x), len(test_y))
  print(sum([len(seq) for seq in train_y]), sum([len(seq) for seq in valid_y]), sum([len(seq) for seq in test_y]))
  emb_layer = modules.EmbeddingLayer(
    args.d, data, fix_emb = False,
    embs = dataloader.load_embedding(args.word_embedding))

  print(len(emb_layer.word2id))
      
  nclasses = len(label_to_ix)

  train_x, train_y = dataloader.create_batches(
    train_x, train_y,
    args.batch_size,
    emb_layer.word2id, 
  )
  valid_x, valid_y = dataloader.create_batches(
    valid_x, valid_y,
    args.batch_size,
    emb_layer.word2id, 
  )
  test_x, test_y = dataloader.create_batches(
    test_x, test_y,
    args.batch_size,
    emb_layer.word2id, 
  )

  model = Model(args, emb_layer, nclasses).cuda()
  
  need_grad = lambda x: x.requires_grad
  if args.adam:
    optimizer = optim.Adam(
      model.parameters(),
      #filter(need_grad, model.parameters()),
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
    best_valid, test_result = train_model(epoch, model, optimizer,
      train_x, train_y,
      valid_x, valid_y,
      test_x, test_y,
      best_valid, test_result
    )
    if args.lr_decay>0:
      optimizer.param_groups[0]['lr'] *= args.lr_decay
    print(model.eval_time)
    print(model.emb_time)
    print(model.classify_time)

  sys.stdout.write("best_valid: {:.6f}\n".format(
    best_valid
  ))
  sys.stdout.write("test_err: {:.6f}\n".format(
    test_result
  ))


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
  argparser.add_argument("--gcnn", action='store_true', help="whether to use gated cnn")
  argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
  argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
  argparser.add_argument("--dilated", action='store_true', help="whether to use dialted CNN")
  argparser.add_argument("--sru", action='store_true', help="whether to use SRU")
  argparser.add_argument("--adam", action='store_true', help="whether to use adam")
  argparser.add_argument("--path", type=str, required=True, help="path to corpus directory")
  argparser.add_argument("--word_embedding", type=str, required=True, help="word vectors")
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
