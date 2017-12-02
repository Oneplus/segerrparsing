#!/usr/bin/env python
import sys
import itertools
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def flatten(lst):
  return list(itertools.chain.from_iterable(lst))


def deep_iter(x):
  if isinstance(x, list) or isinstance(x, tuple):
    for u in x:
      for v in deep_iter(u):
        yield v
  else:
    yield x


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

  def _get_indices(self, y):
    indices = []
    max_len = len(y[0])
    for i in range(len(y)):
      cur_len = len(y[i])
      indices += [i * max_len + x for x in range(cur_len)]
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
      tag_scores = F.log_softmax(tag_scores)
    _, tag_result = torch.max(tag_scores, 1)

    if self.training:
      return self._get_tag_list(tag_result.view(1, -1), y), F.nll_loss(tag_scores, tag_vec, size_average=False)
    else:
      return self._get_tag_list(tag_result.view(1, -1), y), torch.FloatTensor([0.0])


class EmbeddingLayer(nn.Module):
  def __init__(self, n_d, words, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
    super(EmbeddingLayer, self).__init__()
    word2id = {}
    if embs is not None:
      embwords, embvecs = embs
      for word in embwords:
        assert word not in word2id, "Duplicate words in pre-trained embeddings"
        word2id[word] = len(word2id)

      logging.info("{} pre-trained word embeddings loaded.".format(len(word2id)))
      if n_d != len(embvecs[0]):
        logging.warning("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
          n_d, len(embvecs[0]), len(embvecs[0])
          ))
        n_d = len(embvecs[0])

    for w in deep_iter(words):
      if w not in word2id:
        word2id[w] = len(word2id)

    if oov not in word2id:
      word2id[oov] = len(word2id)

    if pad not in word2id:
      word2id[pad] = len(word2id)

    self.word2id = word2id
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
      norms = weight.data.norm(2,1)
      if norms.dim() == 1:
        norms = norms.unsqueeze(1)
      weight.data.div_(norms.expand_as(weight.data))

    if fix_emb:
      self.embedding.weight.requires_grad = False

  def forward(self, input):
    return self.embedding(input)
