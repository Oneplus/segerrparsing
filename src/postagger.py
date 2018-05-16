#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import tempfile
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from seqlabel.modules import MultiLayerCNN, GatedCNN, DilatedCNN, \
  ClassifyLayer, EmbeddingLayer, PartialClassifyLayer, CRFLayer, PartialCRFLayer
from seqlabel.dataloader import load_embedding
from seqlabel.utils import deep_iter, dict2namedtuple
import subprocess
try:
  import seqlabel.cuda_functional as MF
except:
  print('SRU is not supported.', file=sys.stderr)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def read_corpus(path):
  """
  read CoNLL format data.

  :param path:
  :return:
  """
  dataset = []
  labels_dataset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for lines in fin.read().strip().split('\n\n'):
      data, labels = [], []
      for line in lines.splitlines():
        if line[0] == '#':
          continue
        tokens = line.split()
        if tokens[0].find('.') != -1 or tokens[0].find('-') != -1:
          continue
        if len(tokens) == 6:
          tokens.insert(1, '\u3000')
          tokens.insert(2, '\u3000')
        data.append(tokens[1])
        labels.append(tokens[3])
      dataset.append(data)
      labels_dataset.append(labels)
  return dataset, labels_dataset

def read_data(train_path, valid_path, test_path):
  train_x, train_y = read_corpus(train_path)
  valid_x, valid_y = read_corpus(valid_path)
  test_x, test_y = read_corpus(test_path) if test_path is not None else ([], [])
  return train_x, train_y, valid_x, valid_y, test_x, test_y

def read_elmo(elmo_path, x):
  try:
    f = h5py.File(elmo_path, 'r')
    elmo = []
    for x_i in x:
      sent = ' '.join(x_i)
      #elmo.append(np.average(f[sent].value, axis=0))
      elmo.append(f[sent].value)
      assert len(x_i) == elmo[-1].shape[-2]
    f.close()
  except:
    logging.info('ELMo file doesn\'t exist.')
    exit(0)

  return elmo

def read_cluster(cluster_path):
  cluster2id = {}
  cluster_lexicon = {'<oov>':0, '<pad>':1}
  with codecs.open(cluster_path, 'r', encoding='utf-8') as fin:
    for line in fin.read().strip().split('\n'):
      items = line.split('\t')
      if items[1] not in cluster_lexicon: 
        if items[0] not in cluster2id:
          cluster2id[items[0]] = len(cluster2id)
        cluster_lexicon[items[1]] = cluster2id[items[0]] + 2
  return cluster_lexicon

def create_one_batch(x, e, y, word2id, char2id, cluster2id, oov='<oov>', pad='<pad>', sort=True, use_cuda=False):
  batch_size = len(x)
  lst = list(range(batch_size))
  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  if e is not None:
    e = [e[i] for i in lst]
  y = [y[i] for i in lst]
  lens = [len(x[i]) for i in lst]
  max_len = max(lens)

  oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
  assert oov_id is not None and pad_id is not None
  batch_x = torch.LongTensor(batch_size, max_len).fill_(pad_id)
  for i, x_i in enumerate(x):
    for j, x_ij in enumerate(x_i):
      batch_x[i][j] = word2id.get(x_ij.lower(), oov_id)

  max_chars = max([len(w) for i in lst for w in x[i]]) + 1  # counting the <bos>
  bos_id, oov_id, pad_id = char2id.get('<bos>', None), char2id.get(oov, None), char2id.get(pad, None)
  assert bos_id is not None and oov_id is not None and pad_id is not None
  batch_c = torch.LongTensor(batch_size * max_len, max_chars).fill_(pad_id)
  batch_c_lens = torch.LongTensor(batch_size * max_len).fill_(1)
  for i, x_i in enumerate(x):
    for j, x_ij in enumerate(x_i):
      batch_c_lens[i * max_len + j] = len(x_ij) + 1  # counting the <bos>
  new_batch_c_lens, indices_t = torch.sort(batch_c_lens, descending=True)

  indices = torch.LongTensor(batch_size * max_len)
  for i in range(batch_size * max_len):
    indices[indices_t[i]] = i

  for t, idx in enumerate(indices):
    i, j = int(idx) // max_len, int(idx) % max_len
    batch_c[t][0] = bos_id
    if j < len(x[i]):
      x_ij = x[i][j]
      for k, c in enumerate(x_ij):
        batch_c[t][k + 1] = char2id.get(c.lower(), oov_id)

  if e is not None:
    batch_e = torch.FloatTensor(batch_size, max_len, e[0].shape[-1]).fill_(0)
    for i, x_i in enumerate(x):
      assert len(x_i) == e[i].shape[-2]
      for j, x_ij in enumerate(x_i):
        batch_e[i][j] = torch.from_numpy(e[i][j])
  else:
    batch_e = None


  if cluster2id is not None:
    oov_id, pad_id = cluster2id.get(oov, None), cluster2id.get(pad, None)
    assert oov_id is not None and pad_id is not None
    batch_cluster = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_cluster[i][j] = cluster2id.get(x_ij.lower(), oov_id)
  else:
    batch_cluster = None
    
  batch_y = torch.LongTensor(batch_size, max_len).fill_(0)
  for i, y_i in enumerate(y):
    for j, y_ij in enumerate(y_i):
      batch_y[i][j] = y_ij
  if use_cuda:
    batch_x = batch_x.cuda()
    batch_c = batch_c.cuda()
    batch_y = batch_y.cuda()
    if batch_cluster is not None:
      batch_cluster = batch_cluster.cuda()
    indices = indices.cuda()

  return batch_x, (batch_c, new_batch_c_lens.tolist(), indices), batch_e, batch_cluster, batch_y, lens


# shuffle training examples and create mini-batches
def create_batches(x, e, y, batch_size, word2id, char2id, cluster2id, perm=None, shuffle=True, sort=True, use_cuda=False, text=None):
  lst = perm or list(range(len(x)))
  if shuffle:
    random.shuffle(lst)

  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  y = [y[i] for i in lst]
  if e is not None:
    e = [e[i] for i in lst]
  if text is not None:
    text = [text[i] for i in lst]

  sum_len = 0.0
  batches_x, batches_c, batches_e, batches_cluster, batches_y, batches_lens, batches_text = [], [], [], [], [], [], []
  size = batch_size
  nbatch = (len(x) - 1) // size + 1
  for i in range(nbatch):
    start_id, end_id = i * size, (i + 1) * size
    bx, bc, be, bcluster, by, blens = create_one_batch(x[start_id: end_id], None if e is None else e[start_id: end_id], y[start_id: end_id], 
                                             word2id, char2id, cluster2id, sort=sort, use_cuda=use_cuda)
    sum_len += sum(blens)
    batches_x.append(bx)
    batches_c.append(bc)
    batches_e.append(be)
    batches_cluster.append(bcluster)
    batches_y.append(by)
    batches_lens.append(blens)
    
    if text is not None:
      batches_text.append(text[start_id: end_id])

  if sort:
    perm = list(range(nbatch))
    random.shuffle(perm)
    batches_x = [batches_x[i] for i in perm]
    batches_c = [batches_c[i] for i in perm]
    batches_y = [batches_y[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    if e is not None:
      batches_e = [batches_e[i] for i in perm]
    if text is not None:
      batches_text = [batches_text[i] for i in perm]
    if cluster2id is not None:
      batches_cluster = [batches_cluster[i] for i in perm]

  logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
  if text is not None:
    return batches_x, batches_c, batches_e, batches_cluster, batches_y, batches_lens, batches_text
  return batches_x, batches_c, batches_e, batches_cluster, batches_y, batches_lens

class EnsembleModel(nn.Module):
  def __init__(self, models):
    super(EnsembleModel, self).__init__()
    self.n_models = len(models)
    self.models = models

  def forward(self, word_inp, chars_package, elmo, cluster_inp, y):
    output = self.models[0].encoder_output(word_inp, chars_package, elmo, cluster_inp)
    probs = self.models[0].classify_layer.get_probs(output)
    for model in self.models[1:]:
      output = model.encoder_output(word_inp, chars_package, elmo, cluster_inp)
      probs.add_(model.classify_layer.get_probs(output))
    probs.div_(self.n_models)

    _, tag_result = torch.max(probs[:, :, 1:], 2)
    tag_result.add_(1)

    return tag_result, torch.FloatTensor([0.0])    

class Model(nn.Module):
  def __init__(self, opt, word_emb_layer, char_emb_layer, cluster_emb_layer, n_class, use_cuda):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.opt = opt
    self.use_partial = opt.use_partial
    self.word_emb_layer = word_emb_layer
    self.char_emb_layer = char_emb_layer
    self.cluster_emb_layer = cluster_emb_layer

    self.char_lstm = nn.LSTM(char_emb_layer.n_d, char_emb_layer.n_d * 2, 
                             num_layers=1, bidirectional=True,
                             batch_first=True, dropout=opt.dropout)

    encoder_output = None
    
    encoder_input = word_emb_layer.n_d + 4 * char_emb_layer.n_d
    if cluster_emb_layer is not None:
      encoder_input += cluster_emb_layer.n_d
    #self.merge_inputs = nn.Linear(encoder_input, opt.hidden_dim)
    if opt.encoder.lower() == 'lstm':
      self.encoder = nn.LSTM(encoder_input + opt.elmo_dim if opt.use_elmo else encoder_input, 
                             opt.hidden_dim, num_layers=opt.depth, bidirectional=True,
                             batch_first=True, dropout=opt.dropout)
      encoder_output = opt.hidden_dim * 2

    if opt.classifier.lower() == 'vanilla':
      classify_layer_func = PartialClassifyLayer if self.use_partial else ClassifyLayer
      self.classify_layer = classify_layer_func(encoder_output, n_class, self.use_cuda)
    elif opt.classifier.lower() == 'crf':
      classify_layer_func = PartialCRFLayer if self.use_partial else CRFLayer
      self.classify_layer = classify_layer_func(encoder_output, n_class, self.use_cuda)
    else:
      raise ValueError('Unknown classifier {0}'.format(opt.classifier))

    self.train_time = 0
    self.eval_time = 0
    self.emb_time = 0
    self.classify_time = 0

  def forward(self, word_inp, chars_package, elmo, cluster_inp, y):
    output = self.encoder_output(word_inp, chars_package, elmo, cluster_inp)
    #output = F.dropout(output, 0.5, self.training)

    start_time = time.time()

    output, loss = self.classify_layer.forward(output, y)

    if not self.training:
      self.classify_time += time.time() - start_time
    if self.training:
      loss += self.opt.l2 * self.classify_layer.hidden2tag.weight.data.norm(2)
      #loss += self.opt.l2 * self.merge_inputs.weight.data.norm(2)
    return output, loss

  def encoder_output(self, word_inp, chars_package, elmo, cluster_inp):
    start_time = time.time()
    batch_size, seq_len = word_inp.size(0), word_inp.size(1)
    word_emb = self.word_emb_layer(Variable(word_inp).cuda() if self.use_cuda else Variable(word_inp))

    chars_inp, chars_lengths, chars_real_indices = chars_package
    chars_emb = self.char_emb_layer(Variable(chars_inp).cuda() if self.use_cuda else Variable(chars_inp))
    packed_chars_emb = nn.utils.rnn.pack_padded_sequence(chars_emb, chars_lengths, batch_first=True)
    _, (chars_outputs, __) = self.char_lstm(packed_chars_emb)
    chars_outputs = chars_outputs.permute(1, 0, 2).contiguous().view(-1, self.opt.char_dim * 4)
    chars_outputs = chars_outputs[chars_real_indices, :].view(-1, seq_len, self.opt.char_dim * 4)

    emb = torch.cat([word_emb, chars_outputs], dim=2)
    if elmo is not None:   
      elmo = Variable(elmo).cuda() if self.use_cuda else Variable(elmo)   
      emb = torch.cat([emb, elmo], dim=2)

    if self.opt.use_cluster:
      cluster_emb = self.cluster_emb_layer(Variable(cluster_inp).cuda() if self.use_cuda else Variable(cluster_inp))
      emb = torch.cat([emb, cluster_emb], dim = 2)

    emb = F.dropout(emb, self.opt.dropout, self.training)
    
    if not self.training:
      self.emb_time += time.time() - start_time

    start_time = time.time()

    if self.opt.encoder.lower() in ('lstm', 'sru'):
      # x = emb.permute(1, 0, 2)  -- for SRU
      output, hidden = self.encoder(emb)
      # output = output.permute(1, 0, 2)
    elif self.opt.encoder.lower() in ('cnn', 'gcnn'):
      output = self.encoder(emb)
    elif self.opt.encoder.lower() == 'dilated':
      output = self.encoder(emb)
    else:
      raise ValueError('unknown encoder: {0}'.format(self.opt.encoder))
    if self.training:
      self.train_time += time.time() - start_time
    else:
      self.eval_time += time.time() - start_time

    return output

def eval_model(model, valid, ix2label, args, gold_path):
  if args.output is not None:
    path = args.output
    fpo = codecs.open(path, 'w', encoding='utf-8')
  else:
    descriptor, path = tempfile.mkstemp(suffix='.tmp')
    fpo = codecs.getwriter('utf-8')(os.fdopen(descriptor, 'w'))

  valid_x, valid_c, valid_e, valid_cluster, valid_y, valid_lens, valid_text = valid

  model.eval()
  for x, c, e, cluster, y, lens, text in zip(valid_x, valid_c, valid_e, valid_cluster, valid_y, valid_lens, valid_text):
    output, loss = model.forward(x, c, e, cluster, y)
    output_data = output.data
    for bid in range(len(x)):
      for k, (word, tag) in enumerate(zip(text[bid], output_data[bid])):
        tag = ix2label[int(tag)]
        print('{0}\t{1}\t{1}\t{2}\t{2}\t_\t_\t_'.format(k + 1, word, tag), file=fpo)
      print(file=fpo)
  fpo.close()
  model.train()
  p = subprocess.Popen([args.script, gold_path, path], stdout=subprocess.PIPE)
  p.wait()
  f = 0
  for line in p.stdout.readlines():
    f = line.strip().split()[-1]
  os.remove(path)
  return float(f)


def train_model(epoch, model, optimizer,
                train, valid, test, ix2label, best_valid, test_result):
  model.train()
  opt = model.opt

  total_loss, total_tag = 0.0, 0
  cnt = 0
  start_time = time.time()

  train_x, train_c, train_e, train_cluster, train_y, train_lens = train

  lst = list(range(len(train_x)))
  random.shuffle(lst)
  train_x = [train_x[l] for l in lst]
  train_c = [train_c[l] for l in lst]
  train_e = [train_e[l] for l in lst]
  train_cluster = [train_cluster[l] for l in lst]
  train_y = [train_y[l] for l in lst]
  train_lens = [train_lens[l] for l in lst]

  for x, c, e, cluster, y, lens in zip(train_x, train_c, train_e, train_cluster, train_y, train_lens):
    cnt += 1
    model.zero_grad()
    _, loss = model.forward(x, c, e, cluster, y)
    total_loss += loss.data[0]
    n_tags = sum(lens)
    total_tag += n_tags
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip_grad)
    optimizer.step()
    if cnt * opt.batch_size % 1024 == 0:
      logging.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={:.2f}s".format(
        epoch, cnt, optimizer.param_groups[0]['lr'],
        1.0 * loss.data[0] / n_tags, time.time() - start_time
      ))
      start_time = time.time()

    if cnt % opt.eval_steps == 0:
      valid_result = eval_model(model, valid, ix2label, opt, opt.gold_valid_path)
      logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}".format(
        epoch, cnt, optimizer.param_groups[0]['lr'], total_loss, valid_result))

      if valid_result > best_valid:
        torch.save(model.state_dict(), os.path.join(opt.model, 'model.pkl'))
        logging.info("New record achieved!")
        best_valid = valid_result
        if test is not None:
          test_result = eval_model(model, test, ix2label, opt, opt.gold_test_path)          
          logging.info("Epoch={} iter={} lr={:.6f} test_acc={:.6f}".format(
            epoch, cnt, optimizer.param_groups[0]['lr'], test_result))    

  return best_valid, test_result

def label_to_index(y, label_to_ix, incremental=True):
  for i in range(len(y)):
    for j in range(len(y[i])):
      if y[i][j] not in label_to_ix and incremental:
        label = label_to_ix[y[i][j]] = len(label_to_ix)
      else:
        label = label_to_ix.get(y[i][j], 0)
      y[i][j] = label

def train():
  cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
  cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
  cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
  cmd.add_argument('--encoder', default='lstm', choices=['lstm'],
                   help='the type of encoder: valid options=[lstm]')
  cmd.add_argument('--classifier', default='vanilla', choices=['vanilla', 'crf'],
                   help='The type of classifier: valid options=[vanilla, crf]')
  cmd.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'],
                   help='the type of optimizer: valid options=[sgd, adam]')
  cmd.add_argument('--train_path', required=True, help='the path to the training file.')
  cmd.add_argument('--valid_path', required=True, help='the path to the validation file.')
  cmd.add_argument('--test_path', required=False, help='the path to the testing file.')

  cmd.add_argument('--gold_valid_path', type=str, help='the path to the validation file.')
  cmd.add_argument('--gold_test_path', type=str, help='the path to the testing file.')

  cmd.add_argument('--use_elmo', action= 'store_true', help='whether to use elmo.')
  cmd.add_argument('--train_elmo_path', type=str, help='the path to the train elmo.')
  cmd.add_argument('--valid_elmo_path', type=str, help='the path to the validation elmo.')
  cmd.add_argument('--test_elmo_path', type=str, help='the path to the testing elmo.')

  cmd.add_argument('--use_cluster', action= 'store_true', help='whether to use brown cluster.')
  cmd.add_argument('--cluster_path', type=str, help='the path to the brown cluster.')

  cmd.add_argument("--model", required=True, help="path to save model")
  cmd.add_argument("--word_embedding", type=str, required=True, help="word vectors")
  cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
  cmd.add_argument("--hidden_dim", "--hidden", type=int, default=128, help='the hidden dimension.')
  cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
  cmd.add_argument("--elmo_dim", type=int, default=1024, help='the elmo dimension.')
  cmd.add_argument("--cluster_dim", type=int, default=50, help='the brown cluster dimension.')
  cmd.add_argument("--word_dim", type=int, default=100, help='the input dimension.')
  cmd.add_argument("--char_dim", type=int, default=50, help='the char dimension.')
  cmd.add_argument("--dropout", type=float, default=0.0, help='the dropout rate')
  cmd.add_argument("--depth", type=int, default=2, help='the depth of lstm')

  cmd.add_argument("--eval_steps", type=int, help='eval every x batches')
  cmd.add_argument("--l2", type=float, default=0.00001, help='the l2 decay rate.')
  cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
  cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
  cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')
  cmd.add_argument("--use_partial", default=False, action='store_true', help="whether use the partial data")
  cmd.add_argument('--output', help='The path to the output file.')
  cmd.add_argument("--script", required=True, help="The path to the evaluation script")

  opt = cmd.parse_args(sys.argv[2:])

  print(opt)
  torch.manual_seed(opt.seed)
  random.seed(opt.seed)
  if opt.gpu >= 0:
    torch.cuda.set_device(opt.gpu)
    if opt.seed > 0:
      torch.cuda.manual_seed(opt.seed)

  if opt.gold_valid_path is None:
    opt.gold_valid_path = opt.valid_path

  if opt.gold_test_path is None and opt.test_path is not None:
    opt.gold_test_path = opt.test_path

  use_cuda = opt.gpu >= 0 and torch.cuda.is_available()

  train_x, train_y, valid_x, valid_y, test_x, test_y = read_data(
    opt.train_path, opt.valid_path, opt.test_path)
  logging.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
    len(train_y), len(valid_y), len(test_y)))
  logging.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
    sum([len(seq) for seq in train_y]), sum([len(seq) for seq in valid_y]), sum([len(seq) for seq in test_y])))

  if opt.use_partial:
      label_to_ix = {'<pad>': 0, '<UNK>': 1}
  else:
      label_to_ix = {'<pad>': 0}

  label_to_index(train_y, label_to_ix)
  label_to_index(valid_y, label_to_ix, incremental=False)
  label_to_index(test_y, label_to_ix, incremental=False)
  logging.info('number of tags: {0}'.format(len(label_to_ix)))

  embs_words, embs = load_embedding(opt.word_embedding)
  word_lexicon = {word: i for i, word in enumerate(embs_words)}
  logging.info('Pretrained word embedding size: {0}'.format(len(word_lexicon)))

  char_lexicon = {}

  for x in train_x:
    for w in x:
      pass
      #if w.lower() not in word_lexicon:        
      #  word_lexicon[w.lower()] = len(word_lexicon)
      for ch in w:
        if ch.lower() not in char_lexicon:
          char_lexicon[ch.lower()] = len(char_lexicon)

  for special_word in ['<oov>', '<pad>']:
    if special_word not in word_lexicon:
      word_lexicon[special_word] = len(word_lexicon)

  for special_char in ['<bos>', '<oov>', '<pad>']:
    if special_char not in char_lexicon:
      char_lexicon[special_char] = len(char_lexicon)

  if opt.use_cluster:
    if opt.cluster_path is None:
      logging.info('need cluster path')

    logging.info('Reading cluster file.')
    cluster_lexicon = read_cluster(opt.cluster_path)
    cluster_emb_layer = EmbeddingLayer(opt.cluster_dim, cluster_lexicon, fix_emb=False)
    logging.info('size of cluster map: {0}, Cluster embedding size: {1}.'.format(len(cluster_lexicon), cluster_emb_layer.n_V))
  else:
    cluster_emb_layer = None

  word_emb_layer = EmbeddingLayer(opt.word_dim, word_lexicon, fix_emb=False, embs=(embs_words, embs))
  char_emb_layer = EmbeddingLayer(opt.char_dim, char_lexicon, fix_emb=False)

  logging.info('Word embedding size: {0}'.format(len(word_emb_layer.word2id)))
  logging.info('Char embedding size: {0}'.format(len(char_emb_layer.word2id)))

  nclasses = len(label_to_ix)
  ix2label = {ix: label for label, ix in label_to_ix.items()}

  if opt.use_elmo:
    if opt.train_elmo_path is None or opt.valid_elmo_path is None or opt.test_elmo_path is None:
      logging.info('need elmo path for all dataset.')
      exit(0)

    logging.info('Reading ELMo of training dataset.')
    train_e = read_elmo(opt.train_elmo_path, train_x)
    logging.info('Reading ELMo of valid dataset.')
    valid_e = read_elmo(opt.valid_elmo_path, valid_x)
    logging.info('Reading ELMo of testing dataset.')
    test_e = read_elmo(opt.test_elmo_path, test_x)
  else:
    train_e, valid_e, test_e = None, None, None

  word2id, char2id = word_emb_layer.word2id, char_emb_layer.word2id
  cluster2id = cluster_emb_layer.word2id if cluster_emb_layer is not None else None

  train = create_batches(
    train_x, train_e, train_y, opt.batch_size, word2id, char2id, cluster2id, use_cuda=use_cuda)

  if opt.eval_steps is None or opt.eval_steps > len(train_x):
    opt.eval_steps = len(train_x)

  valid = create_batches(
    valid_x, valid_e, valid_y, 1, word2id, char2id, cluster2id, shuffle=False, sort=False, use_cuda=use_cuda, text=valid_x)

  if opt.test_path is not None:
    test = create_batches(
      test_x, test_e, test_y, 1, word2id, char2id, cluster2id, shuffle=False, sort=False, use_cuda=use_cuda, text=test_x)
  else:
    test = None

  model = Model(opt, word_emb_layer, char_emb_layer, cluster_emb_layer, nclasses, use_cuda)

  logging.info(str(model))
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

  with codecs.open(os.path.join(opt.model, 'char.dic'), 'w', encoding='utf-8') as fpo:
    for ch, i in char_emb_layer.word2id.items():
      print('{0}\t{1}'.format(ch, i), file=fpo)

  with codecs.open(os.path.join(opt.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
    for w, i in word_emb_layer.word2id.items():
      print('{0}\t{1}'.format(w, i), file=fpo)

  if cluster_emb_layer is not None:
    with codecs.open(os.path.join(opt.model, 'cluster.dic'), 'w', encoding='utf-8') as fpo:
      for w, i in cluster_emb_layer.word2id.items():
        print('{0}\t{1}'.format(w, i), file=fpo)    

  with codecs.open(os.path.join(opt.model, 'label.dic'), 'w', encoding='utf-8') as fpo:
    for label, i in label_to_ix.items():
      print('{0}\t{1}'.format(label, i), file=fpo)

  json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))
  best_valid, test_result = -1e8, -1e8
  for epoch in range(opt.max_epoch):
    best_valid, test_result = train_model(epoch, model, optimizer,
                                          train, valid, test, ix2label, best_valid, test_result)
    if opt.lr_decay > 0:
      optimizer.param_groups[0]['lr'] *= opt.lr_decay
    logging.info('Total encoder time: {:.2f}s'.format(model.eval_time / (epoch + 1)))
    logging.info('Total embedding time: {:.2f}s'.format(model.emb_time / (epoch + 1)))
    logging.info('Total classify time: {:.2f}s'.format(model.classify_time / (epoch + 1)))

  if opt.use_partial:
      logging.info("best_valid_f: {:.6f}".format(best_valid))
      logging.info("test_f: {:.6f}".format(test_result))
  else:
      logging.info("best_valid_acc: {:.6f}".format(best_valid))
      logging.info("test_acc: {:.6f}".format(test_result))


def test():
  cmd = argparse.ArgumentParser('The testing components of')
  cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
  cmd.add_argument("--input", help="the path to the test file.")
  cmd.add_argument('--output', help='the path to the output file.')
  cmd.add_argument("--models", required=True, help="path to save model")

  cmd.add_argument('--use_elmo', action= 'store_true', help='whether to use elmo.')
  cmd.add_argument('--test_elmo_path', type=str, help='the path to the testing elmo.')

  args = cmd.parse_args(sys.argv[2:])

  if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
  
  models_path = args.models.split(',') 

  print(models_path)

  args2 = dict2namedtuple(json.load(codecs.open(os.path.join(models_path[0], 'config.json'), 'r', encoding='utf-8')))

  char_lexicon = {}
  char_emb_layers = []
  with codecs.open(os.path.join(models_path[0], 'char.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      tokens = line.strip().split('\t')
      if len(tokens) == 1:
        tokens.insert(0, '\u3000')
      token, i = tokens
      char_lexicon[token] = int(i)
  for path in models_path:
    char_emb_layers.append(EmbeddingLayer(args2.char_dim, char_lexicon, fix_emb=False, embs=None))

  word_lexicon = {}
  word_emb_layers = []
  with codecs.open(os.path.join(models_path[0], 'word.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      tokens = line.strip().split('\t')
      if len(tokens) == 1:
        tokens.insert(0, '\u3000') 
      token, i = tokens
      word_lexicon[token] = int(i)
  for path in models_path:
    word_emb_layers.append(EmbeddingLayer(args2.word_dim, word_lexicon, fix_emb=False, embs=None))

  logging.info('word embedding size: ' + str(len(word_emb_layers[0].word2id)))

  if args2.use_cluster:
    cluster_lexicon = {}
    cluster_emb_layers = []
    with codecs.open(os.path.join(models_path[0], 'cluster.dic'), 'r', encoding='utf-8') as fpi:
      for line in fpi:
        token, i = line.strip().split('\t')
        cluster_lexicon[token] = int(i)
    for path in models_path:
      cluster_emb_layers.append(EmbeddingLayer(args2.cluster_dim, cluster_lexicon, fix_emb=False, embs=None))
    logging.info('size of cluster map: {0}'.format(len(cluster_lexicon)))
  else:
    cluster_lexicon = None
    cluster_emb_layers = [None] * len(models_path)

  label2id, id2label = {}, {}
  with codecs.open(os.path.join(models_path[0], 'label.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      token, i = line.strip().split('\t')
      label2id[token] = int(i)
      id2label[int(i)] = token
  logging.info('number of labels: {0}'.format(len(label2id)))

  use_cuda = args.gpu >=0 and torch.cuda.is_available()
  
  models = []

  for idx, path in enumerate(models_path):
    models.append(Model(args2, word_emb_layers[idx], char_emb_layers[idx], cluster_emb_layers[idx], len(label2id), use_cuda))
    models[-1].load_state_dict(torch.load(os.path.join(path, 'model.pkl'), map_location=lambda storage, loc: storage))
    if use_cuda:
      models[-1] = models[-1].cuda()
    
  test_x, test_y = read_corpus(args.input)
  label_to_index(test_y, label2id, incremental=False)

  if args.use_elmo:
    if args.test_elmo_path is None:
      logging.info('need elmo path for testing dataset.')
      exit(0)
    logging.info('Reading ELMo of testing dataset.')
    test_e = read_elmo(args.test_elmo_path, test_x)
  else:
    test_e = None

  test_x, test_c, test_e, test_cluster, test_y, test_lens, test_text = create_batches(
    test_x, test_e, test_y, 1, word_lexicon, char_lexicon, cluster_lexicon, shuffle=False, sort=False, use_cuda=use_cuda, text=test_x)

  if args.output is not None:
    fpo = codecs.open(args.output, 'w', encoding='utf-8')
  else:
    fpo = codecs.getwriter('utf-8')(sys.stdout)

  for model in models:
    model.eval()

  ensemble_model = EnsembleModel(models)

  ensemble_model.eval()

  for x, c, e, cluster, y, lens, text in zip(test_x, test_c, test_e, test_cluster, test_y, test_lens, test_text):
    output, loss = ensemble_model.forward(x, c, e, cluster, y)
    output_data = output.data
    for bid in range(len(x)):
      for k, (word, tag) in enumerate(zip(text[bid], output_data[bid])):
        tag = id2label[int(tag)]
        print('{0}\t{1}\t{1}\t{2}\t{2}\t_\t_\t_'.format(k + 1, word, tag), file=fpo)
      print(file=fpo)
  fpo.close()

if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train()
  elif len(sys.argv) > 1 and sys.argv[1] == 'test':
    test()
  else:
    print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
