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
  def __init__(self, n_in, num_tags, use_cuda=False):
    super(ClassifyLayer, self).__init__()
    self.hidden2tag = nn.Linear(n_in, num_tags)
    self.num_tags = num_tags
    self.use_cuda = use_cuda
    self.logsoftmax = nn.LogSoftmax(dim=2)
    weights = torch.ones(num_tags)
    weights[0] = 0
    self.criterion = nn.NLLLoss(weights)

  def forward(self, x, y):
    """

    :param x: torch.Tensor (batch_size, seq_len, n_in)
    :param y: torch.Tensor (batch_size, seq_len)
    :return:
    """
    tag_scores = self.hidden2tag(x)
    if self.training:
      tag_scores = self.logsoftmax(tag_scores)

    _, tag_result = torch.max(tag_scores[:, :, 1:], 2)
    tag_result.add_(1)
    if self.training:
      return tag_result, self.criterion(tag_scores.view(-1, self.num_tags), Variable(y).view(-1))
    else:
      return tag_result, torch.FloatTensor([0.0])


class PartialClassifyLayer(ClassifyLayer):
  uncertain = 1

  def __init__(self, n_in, num_tags, use_cuda=False):
    super(PartialClassifyLayer, self).__init__(n_in, num_tags, use_cuda)
    weights = torch.ones(num_tags)
    weights[0] = 0
    weights[self.uncertain] = 0
    self.criterion = nn.NLLLoss(weights)

  def forward(self, x, y):
    tag_scores = self.hidden2tag(x)
    if self.training:
      tag_scores = self.logsoftmax(tag_scores)

    _, tag_result = torch.max(tag_scores[:, :, 2:], 2)
    tag_result.add_(2)
    if self.training:
      new_y = torch.LongTensor([y_ for y_ in y.view(-1) if y_ not in (self.uncertain, 0)])
      new_indices = Variable(torch.LongTensor([i for i, y_ in enumerate(y.view(-1)) if y_ not in (self.uncertain, 0)]))
      tag_scores = torch.index_select(tag_scores.view(-1, self.num_tags), 0, new_indices)
      return tag_result, self.criterion(tag_scores, Variable(new_y))
    else:
      return tag_result, torch.FloatTensor([0.0])


class CRFLayer(nn.Module):
  def __init__(self, n_in, num_tags, use_cuda=False):
    super(CRFLayer, self).__init__()
    self.n_in = n_in
    self.num_tags = num_tags
    self.hidden2tag = nn.Linear(n_in, num_tags)
    self.use_cuda = use_cuda

    self.transitions = nn.Parameter(torch.FloatTensor(num_tags, num_tags))
    torch.nn.init.uniform(self.transitions, -0.1, 0.1)

  def forward(self, x, y):
    emissions = self.hidden2tag(x)
    new_emissions = emissions.permute(1, 0, 2).contiguous()

    if self.training:
      new_y = y.permute(1, 0).contiguous()
      numerator = self._compute_joint_llh(new_emissions, Variable(new_y))
      denominator = self._compute_log_partition_function(new_emissions)
      llh = denominator - numerator
      return None, torch.sum(llh)
    else:
      path = self._viterbi_decode(new_emissions)
      path = path.permute(1, 0)
      return path, None

  def _compute_joint_llh(self, emissions, tags):
    seq_length = emissions.size(0)
    llh = torch.zeros_like(tags[0]).float()
    for i in range(seq_length - 1):
      cur_tag, next_tag = tags[i], tags[i + 1]
      llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1)
      transition_score = self.transitions[cur_tag, next_tag]
      llh += transition_score
    cur_tag = tags[-1]
    llh += emissions[-1].gather(1, cur_tag.view(-1, 1)).squeeze(1)

    return llh

  def _compute_log_partition_function(self, emissions):
    seq_length = emissions.size(0)
    log_prob = torch.zeros_like(emissions[0]) + emissions[0]

    for i in range(1, seq_length):
      broadcast_log_prob = log_prob.unsqueeze(2)  # (batch_size, num_tags, 1)
      broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
      broadcast_emissions = emissions[i].unsqueeze(1)  # (batch_size, 1, num_tags)

      score = broadcast_log_prob + broadcast_transitions + broadcast_emissions
      log_prob = self._log_sum_exp(score, 1)

    return self._log_sum_exp(log_prob, 1)

  def _viterbi_decode(self, emissions):
    seq_length = emissions.size(0)

    # (batch_size, num_tags)
    viterbi_score = torch.zeros_like(emissions[0]) + emissions[0]
    viterbi_path = []
    for i in range(1, seq_length):
      # (batch_size, num_tags, 1)
      broadcast_score = viterbi_score.unsqueeze(2)
      # (1, num_tags, num_tags)
      broadcast_transitions = self.transitions.unsqueeze(0)
      # (batch_size, 1, num_tags)
      broadcast_emission = emissions[i].unsqueeze(1)
      # (batch_size, num_tags, num_tags)
      score = broadcast_score + broadcast_transitions + broadcast_emission
      # (batch_size, num_tags), (batch_size, num_tags)
      best_score, best_path = score.max(1)
      viterbi_score = best_score
      viterbi_path.append(best_path)

    # _, (batch_size, )
    _, best_last_tag = viterbi_score.max(1)
    best_tags = [best_last_tag.view(-1, 1)]
    for path in reversed(viterbi_path):
      # indexing
      best_last_tag = path.gather(1, best_tags[-1])
      best_tags.append(best_last_tag)

    best_tags.reverse()
    return torch.stack(best_tags).squeeze(2)

  @staticmethod
  def _log_sum_exp(tensor, dim):
    # Find the max value along `dim`
    offset, _ = tensor.max(dim)
    # Make offset broadcastable
    broadcast_offset = offset.unsqueeze(dim)
    # Perform log-sum-exp safely
    safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))
    # Add offset back
    return offset + safe_log_sum_exp


class PartialCRFLayer(CRFLayer):
  uncertain = 1
  ninf = -1e8

  def __init__(self, n_in, num_tags, use_cuda=False):
    super(PartialCRFLayer, self).__init__(n_in, num_tags, use_cuda)

  def forward(self, x, y):
    emissions = self.hidden2tag(x)
    new_emissions = emissions.permute(1, 0, 2).contiguous()

    if self.training:
      new_y = y.permute(1, 0).contiguous()
      numerator = self._compute_log_constrained_partition_function(new_emissions, Variable(new_y))
      denominator = self._compute_log_partition_function(new_emissions)
      llh = denominator - numerator
      return None, torch.sum(llh)
    else:
      path = self._viterbi_decode(new_emissions)
      path = path.permute(1, 0)
      return path, None

  def _compute_log_constrained_partition_function(self, emissions, tags):
    seq_length, batch_size = tags.size(0), tags.size(1)
    mask = torch.ones(seq_length, batch_size, self.num_tags).float()

    # create mask
    tags_data = tags.data
    for i in range(seq_length):
      for j in range(batch_size):
        if tags_data[i][j] != self.uncertain:
          mask[i][j].zero_()
          mask[i][j][tags_data[i][j]] = 1
        else:
          mask[i][j][0] = 0
          mask[i][j][self.uncertain] = 0

    mask = Variable(mask)
    log_prob = emissions[0] * mask[0] + (1 - mask[0]) * self.ninf

    for i in range(1, seq_length):
      prev_mask, cur_mask = mask[i - 1], mask[i]
      transition_mask = torch.bmm(prev_mask.unsqueeze(2), cur_mask.unsqueeze(1))
      # (batch_size, num_tags, 1)
      broadcast_log_prob = log_prob.unsqueeze(2) * prev_mask.unsqueeze(2) +\
                           (1 - prev_mask.unsqueeze(2)) * self.ninf
      # (batch_size, num_tags, num_tags)
      broadcast_transitions = self.transitions.unsqueeze(0) * transition_mask +\
                              (1 - transition_mask) * self.ninf
      # (batch_size, 1, num_tags)
      broadcast_emissions = emissions[i].unsqueeze(1) * cur_mask.unsqueeze(1) +\
                            (1 - cur_mask.unsqueeze(1)) * self.ninf

      score = broadcast_log_prob + broadcast_transitions + broadcast_emissions
      log_prob = self._log_sum_exp(score, 1)

    return self._log_sum_exp(log_prob, 1)


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

  def forward(self, input_):
    return self.embedding(input_)
