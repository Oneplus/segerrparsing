#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import flatten, deep_iter
import copy

PARTIAL = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class MultiLayerCNN(nn.Module):
  def __init__(self, n_in, hidden_dim, depth, dropout, use_cuda=True):
    super(MultiLayerCNN, self).__init__()
    self.conv2d_layer = []
    self.conv2d_layer.append(nn.Conv2d(1, hidden_dim, (3, n_in), padding=(1, 0)))
    for i in range(depth - 1):
      self.conv2d_layer.append(nn.Conv2d(1, hidden_dim, (3, hidden_dim), padding=(1, 0)))
    if use_cuda:
      for i in range(depth):
        self.conv2d_layer[i].cuda()
    self.n_in = n_in
    self.dropout = dropout

  def forward(self, x):
    x = x.view(x.size(0), 1, -1, self.n_in)
    for conv in self.conv2d_layer:
      x = F.relu(conv(x))
      x = x.permute(0, 3, 2, 1)
      x = F.dropout(x, self.dropout, self.training)
    return x


class GatedCNN(nn.Module):
  def __init__(self, n_in, hidden_dim, depth, dropout, use_cuda=True):
    super(GatedCNN, self).__init__()
    self.conv2d_W = []
    self.conv2d_V = []
    self.conv2d_W.append(nn.Conv2d(1, hidden_dim, (3, n_in), padding=(1, 0)))
    self.conv2d_V.append(nn.Conv2d(1, hidden_dim, (3, n_in), padding=(1, 0)))
    for i in range(depth - 1):
      self.conv2d_W.append(nn.Conv2d(1, hidden_dim, (3, hidden_dim), padding=(1, 0)))
      self.conv2d_V.append(nn.Conv2d(1, hidden_dim, (3, hidden_dim), padding=(1, 0)))
    if use_cuda:
      for i in range(depth):
        self.conv2d_W[i].cuda()
        self.conv2d_V[i].cuda()
    self.n_in = n_in
    self.dropout = dropout

  def forward(self, x):
    x = x.view(x.size(0), 1, -1, self.n_in)
    for W, V in zip(self.conv2d_W, self.conv2d_V):
      g = F.sigmoid(V(x)).permute(0, 3, 2, 1)
      x = W(x).permute(0, 3, 2, 1)
      x = x * g
      x = F.dropout(x, self.dropout, self.training)
    return x


class DilatedCNN(nn.Module):
  def __init__(self, n_in, hidden_dim, depth, dropout, use_cuda=True):
    super(DilatedCNN, self).__init__()
    self.n_dilated_layer = 5
    self.conv_1 = nn.Conv2d(1, hidden_dim, (1, n_in))
    if use_cuda:
      self.conv_1.cuda()
    self.conv2d_W = []
    for i in range(depth):
      self.conv2d_W.append(nn.Conv2d(1, hidden_dim, (3, hidden_dim), padding=(pow(2, i), 0), dilation=(pow(2, i), 1)))
      if use_cuda:
        self.conv2d_W[-1].cuda()
    self.n_in = n_in
    self.dropout = dropout
    self.depth = depth

  def forward(self, x):
    inputs = [self.conv_1(x.view(x.size(0), 1, -1, self.n_in))]
    feat = inputs[0]
    for i in range(self.depth):
      x = inputs[-1]
      for conv in self.conv2d_W:
        x = F.relu(conv(x)).permute(0, 3, 2, 1)
        x = F.dropout(x, self.dropout, self.training)
      inputs.append(x)
      feat = feat + x
    return feat


class ClassifyLayer(nn.Module):
  def __init__(self, n_in, tag_size, use_cuda=False):
    super(ClassifyLayer, self).__init__()
    self.hidden2tag = nn.Linear(n_in, tag_size)
    self.n_in = n_in
    self.use_cuda = use_cuda
    self.softmax = nn.Softmax(dim=1)

  def _get_indices(self, y):
    indices = []
    max_len = max([len(_) for _ in y])
    for i in range(len(y)):
      cur_len = len(y[i])
      indices += [i * max_len + x for x in range(cur_len)]
    # print("standard_indices {0}".format(indices))
    return indices

  def _get_tag_list(self, tag_result, y):
    tag_list = []
    last = 0
    for i in range(len(y)):
      tag_list.append(tag_result[0][last: last + len(y[i])].data.tolist())
      last += len(y[i])
    return tag_list

  def forward(self, x, y):
    tag_vec = Variable(torch.LongTensor(flatten(y))).cuda() if self.use_cuda \
      else Variable(torch.LongTensor(flatten(y)))
    indices = Variable(torch.LongTensor(self._get_indices(y))).cuda() if self.use_cuda \
      else Variable(torch.LongTensor(self._get_indices(y)))
    tag_scores = self.hidden2tag(torch.index_select(x.contiguous().view(-1, self.n_in), 0, indices))
    if self.training:
      tag_scores = self.softmax(tag_scores)

    _, tag_result = torch.max(tag_scores, 1)
    # print("res_index = {0}".format(tag_result))
    if self.training:
      return self._get_tag_list(tag_result.view(1, -1), y), F.nll_loss(tag_scores, tag_vec, size_average=False)
    else:
      return self._get_tag_list(tag_result.view(1, -1), y), torch.FloatTensor([0.0])


class PartialClassifyLayer(ClassifyLayer):
  def __init__(self, n_in, tag_size,  use_cuda=False):
    super(PartialClassifyLayer, self).__init__(n_in, tag_size, use_cuda)

  def _get_indices_except_partical(self, y):
    indices = []
    temp_y = copy.deepcopy(y)
    max_len = max([len(sentence) for sentence in y])
    temp_y = [sentence + [0] * (max_len - len(sentence)) for sentence in temp_y]
    # print("temp_y = {0}".format(temp_y))
    count  = -1
    for index, value in enumerate(temp_y):
      for index_word, value_word in enumerate(value):
        count += 1
        if value_word != PARTIAL and value_word != 0:
          indices.append(count)
    return indices

  def _get_tag_list(self, tag_result, y):
    # y = b
    tag_list = []
    last = 0
    for i in range(len(y)):
      tag_list.append(tag_result[0][last: last + len(y[i])].data.tolist())
      last += len(y[i])
    return tag_list

  def forward(self, x, y):
    temp_y = y
    b = []

    # for i in range(len(temp_y)):
    #   if PARTIAL in temp_y[i]:
    #     temp_y[i].remove(PARTIAL)
    for i in range(len(temp_y)):
      temp = []
      for value in temp_y[i]:
        if value!=PARTIAL:
          temp.append(value)
      b.append(temp)

    tag_vec = Variable(torch.LongTensor(flatten(b))).cuda() if self.use_cuda \
      else Variable(torch.LongTensor(flatten(b)))
    indices_partial = Variable(torch.LongTensor(self._get_indices_except_partical(y))).cuda() if self.use_cuda \
      else Variable(torch.LongTensor(self._get_indices_except_partical(y)))
    indices = Variable(torch.LongTensor(self._get_indices(y))).cuda() if self.use_cuda \
        else Variable(torch.LongTensor(self._get_indices(y)))
    tag_scores_partial = self.hidden2tag(torch.index_select(x.contiguous().view(-1, self.n_in), 0, indices_partial))
    tag_scores = self.hidden2tag(torch.index_select(x.contiguous().view(-1, self.n_in), 0, indices))
    # 这里的tag_scores已经去掉了CIXIN的那些索引了，但是最终的预测是否需要去掉？？
    if self.training:
      tag_scores = F.log_softmax(tag_scores)
      tag_scores_partial = F.log_softmax(tag_scores_partial)

    r, c = tag_scores.size()
    if self.use_cuda:
      a1 = Variable(torch.FloatTensor(r * [10000000])).view(-1, 1).cuda()
      a2 = Variable(torch.ones(r, c - 1)).cuda()
    else:
      a1 = Variable(torch.FloatTensor(r * [10000000])).view(-1, 1)
      a2 = Variable(torch.ones(r, c - 1))
    a3 = torch.cat((a1, a2), 1)
    if self.use_cuda:
      temp = Variable(torch.zeros(r, c)).cuda()
    else:
      temp = Variable(torch.zeros(r, c))
    # print("origin_tag_scores {0}".format(tag_scores.data.tolist()))
    tag_scores = torch.addcmul(temp, 1, a3, tag_scores)

    r1, c1 = tag_scores_partial.size()
    if self.use_cuda:
      a11 = Variable(torch.FloatTensor(r1 * [10000000])).view(-1, 1).cuda()
      a21 = Variable(torch.ones(r1, c1 - 1)).cuda()
    else:
      a11 = Variable(torch.FloatTensor(r1 * [10000000])).view(-1, 1)
      a21 = Variable(torch.ones(r1, c1 - 1))
    a31 = torch.cat((a11, a21), 1)
    if self.use_cuda:
      temp = Variable(torch.zeros(r1, c1)).cuda()
    else:
      temp = Variable(torch.zeros(r1, c1))
    # print("origin_tag_scores {0}".format(tag_scores.data.tolist()))
    tag_scores_partial = torch.addcmul(temp, 1, a31, tag_scores_partial)  # 就是尽量不让它预测0出来

    # print("later_tag_scores {0}".format(tag_scores.data.tolist()))

    _, tag_result_partial = torch.max(tag_scores_partial, 1)
    _, tag_result = torch.max(tag_scores, 1)

    # print("tag_result.size() = {0}, y.size() = {1}".format(tag_result.size(), tag_vec.size()))

    # print("tag_scores_partial = {0}, tag_vec = {1}".format(tag_scores_partial[:50], tag_vec[:50]))
    if self.training:
      return self._get_tag_list(tag_result.view(1, -1), y), F.nll_loss(tag_scores_partial, tag_vec, size_average=False)
    else:
      return self._get_tag_list(tag_result.view(1, -1), y), torch.FloatTensor([0.0])


class EmbeddingLayer(nn.Module):
  def __init__(self, n_d, word2id, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
    super(EmbeddingLayer, self).__init__()
    if embs is not None:
      embwords, embvecs = embs
      # for word in embwords:
      #  assert word not in word2id, "Duplicate words in pre-trained embeddings"
      #  word2id[word] = len(word2id)

      logging.info("{} pre-trained word embeddings loaded.".format(len(word2id)))
      if n_d != len(embvecs[0]):
        logging.warning("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
          n_d, len(embvecs[0]), len(embvecs[0])))
        n_d = len(embvecs[0])

    self.word2id = word2id
    self.id2word = {i: word for word, i in word2id.items()}
    self.n_V, self.n_d = len(word2id), n_d
    self.oovid = word2id[oov]
    self.padid = word2id[pad]
    self.embedding = nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
    self.embedding.weight.data.uniform_(-0.25, 0.25)

    if embs is not None:
      weight = self.embedding.weight
      weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
      logging.info("embedding shape: {}".format(weight.size()))

    if normalize:
      weight = self.embedding.weight
      norms = weight.data.norm(2, 1)
      if norms.dim() == 1:
        norms = norms.unsqueeze(1)
      weight.data.div_(norms.expand_as(weight.data))

    if fix_emb:
      self.embedding.weight.requires_grad = False

  def forward(self, input):
    return self.embedding(input)
