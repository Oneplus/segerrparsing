import gzip
import sys
import random
import codecs
import numpy as np
import torch


def pad(sequences, pad_token='<pad>', pad_left=False):
  """
  input sequences is a list of text sequence [[str]]
  pad each text sequence to the length of the longest

  :param sequences:
  :param pad_token:
  :param pad_left:
  :return:
  """
  # max_len = max(5,max(len(seq) for seq in sequences))
  max_len = max(len(seq) for seq in sequences)
  if pad_left:
    return [[pad_token]*(max_len-len(seq)) + seq for seq in sequences]
  return [seq + [pad_token]*(max_len-len(seq)) for seq in sequences]


def read_corpus(path):
  data = []
  labels = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin:
      terms = line.split()
      data.append([])
      labels.append([])
      for term in terms:
        data[-1].append(term.split('_')[0])
        labels[-1].append(term.split('_')[1].replace('\n', ''))
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
  x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
  x = torch.LongTensor(x)
  assert x.size(0) == length*batch_size
  x = x.view(batch_size, length).t().contiguous()
  if use_cuda:
    x = x.cuda()
  return x, y


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=True, use_cuda=False):
  lst = perm or range(len(x))
  random.shuffle(lst)

  if sort:
    lst = sorted(lst, key=lambda i: -len(y[i]))

  x = [x[i] for i in lst]
  y = [y[i] for i in lst]
    
  sum_len = 0.0
  batches_x = []
  batches_y = []
  size = batch_size
  nbatch = (len(x) - 1) // size + 1
  for i in range(nbatch):
    bx, by = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id, use_cuda=use_cuda)
    sum_len += len(by[0])
    batches_x.append(bx)
    batches_y.append(by)

  if sort:
    perm = range(nbatch)
    random.shuffle(perm)
    batches_x = [batches_x[i] for i in perm]
    batches_y = [batches_y[i] for i in perm]

  sys.stdout.write("{} batches, avg len: {:.1f}\n".format(nbatch, sum_len/nbatch))
  return batches_x, batches_y


def load_embedding_npz(path):
  data = np.load(path)
  return [str(w) for w in data['words']], data['vals']


def load_embedding_txt(path):
  file_open = gzip.open if path.endswith(".gz") else open
  words = []
  vals = []
  with file_open(path) as fin:
    fin.readline()
    for line in fin:
      line = line.strip()
      if line:
        parts = line.split()
        words.append(parts[0].decode('utf-8'))
        vals += [float(x) for x in parts[1:]]
  return words, np.asarray(vals).reshape(len(words), -1)


def load_embedding(path):
  if path.endswith(".npz"):
    return load_embedding_npz(path)
  else:
    return load_embedding_txt(path)