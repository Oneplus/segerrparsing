import gzip
import os
import sys
import re
import random

import numpy as np
import torch

import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

def pad(sequences, pad_token='<pad>', pad_left=False):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    #max_len = max(5,max(len(seq) for seq in sequences))
    max_len = max(len(seq) for seq in sequences)
    if pad_left:
        return [ [pad_token]*(max_len-len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token]*(max_len-len(seq)) for seq in sequences ]

def read_corpus(path):
  tag_to_ix = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
  unigram = []
  bigram = []
  labels = []
  with open(path) as fin:
    for line in fin:
      text, label = line.split('\t')
      unigram.append(text.split())
      
      bigram.append([])
      bigram[-1].append('<s>' + unigram[-1][0])
      for i in range(len(unigram[-1]) - 1):
        bigram[-1].append(unigram[-1][i] + unigram[-1][i + 1])
      bigram[-1].append(unigram[-1][-1] + '</s>')
      labels.append([])
      for x in label.split():
        labels[-1].append(tag_to_ix[x])
  return (unigram, bigram), labels


def read_data(path, seed = 1234):
  train_path = os.path.join(path, "train.txt")
  valid_path = os.path.join(path, "valid.txt")
  test_path = os.path.join(path, "test.txt")
  train_x, train_y = read_corpus(train_path)
  valid_x, valid_y = read_corpus(valid_path)
  test_x, test_y = read_corpus(test_path)
  return train_x, train_y, valid_x, valid_y, test_x, test_y


def create_one_batch(x, y, uni_map2id, bi_map2id, oov='<oov>'):
  lst = range(len(x[0]))
  lst = sorted(lst, key=lambda i: -len(y[i]))

  x = ([x[0][i] for i in lst], [x[1][i] for i in lst])
  y = [ y[i] for i in lst ]    

  oov_id = uni_map2id[oov]
  uni = pad(x[0])
  uni_length = len(uni[0])
  batch_size = len(uni)
  uni = [ uni_map2id.get(w, oov_id) for seq in uni for w in seq ]
  uni = torch.LongTensor(uni)

  assert uni.size(0) == uni_length * batch_size

  oov_id = bi_map2id[oov]
  bi = pad(x[1])
  bi_length = len(bi[0])
  bi = [ bi_map2id.get(w, oov_id) for seq in bi for w in seq ]
  bi = torch.LongTensor(bi)

  assert bi.size(0) == bi_length * batch_size

  return (uni.view(batch_size, uni_length).contiguous(), bi.view(batch_size, bi_length).contiguous()), y

# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, uni_map2id, bi_map2id, perm=None, sort=True):

  lst = perm or range(len(x[0]))
  random.shuffle(lst)

  # sort sequences based on their length; necessary for SST
  if sort:
    lst = sorted(lst, key=lambda i: -len(y[i]))

  x = ([x[0][i] for i in lst], [x[1][i] for i in lst])
  y = [ y[i] for i in lst ]

  sum_len = 0.0
  batches_x = [ ]
  batches_y = [ ]
  size = batch_size
  nbatch = (len(x[0])-1) // size + 1
  for i in range(nbatch):
    bx, by = create_one_batch((x[0][i*size:(i+1)*size], x[1][i*size:(i+1)*size]), y[i*size:(i+1)*size], uni_map2id, bi_map2id)
    sum_len += len(by[0])
    batches_x.append(bx)
    batches_y.append(by)

  if sort:
    perm = range(nbatch)
    random.shuffle(perm)
    batches_x = [ batches_x[i] for i in perm ]
    batches_y = [ batches_y[i] for i in perm ]
  logging.info("{} batches, avg len: {:.1f}".format(
   nbatch, sum_len/nbatch
  ))

  return batches_x, batches_y


def load_embedding_npz(path):
  data = np.load(path)
  return [ str(w) for w in data['words'] ], data['vals']

def load_embedding_txt(path):
  file_open = gzip.open if path.endswith(".gz") else open
  words = [ ]
  vals = [ ]
  with file_open(path) as fin:
    fin.readline()
    for line in fin:
      line = line.strip()
      if line:
        parts = line.split()
        words.append(parts[0])
        vals += [ float(x) for x in parts[1:] ]
  return words, np.asarray(vals).reshape(len(words),-1)

def load_embedding(path):
  if path.endswith(".npz"):
    return load_embedding_npz(path)
  else:
    return load_embedding_txt(path)