#!/usr/bin/env python3

import torch

def pad_sequences(seqs, max_len=None):
  """Matrix of token ids padded with zeros"""

  if max_len is None:
    # set max_len to the length of the longest sequence
    max_len = max(len(id_seq) for id_seq in seqs)

  padded = torch.zeros(len(seqs), max_len, dtype=torch.long)

  for i, seq in enumerate(seqs):
    if len(seq) > max_len:
      seq = seq[:max_len]
    padded[i, :len(seq)] = torch.tensor(seq)

  return padded

def sequences_to_matrix(seqs, n_columns):
  """Convert sequences of indices to multi-hot representations"""
  #print(seqs)
  multi_hot = torch.zeros(len(seqs), n_columns)
  #print(multi_hot.shape)

  for i, seq in enumerate(seqs):
    #print(i)
    #print(seqs)
    #print(i)
    multi_hot[i, seq] = 1.0

  return multi_hot

if __name__ == "__main__":

  seqs = [[1, 2, 3, 4], [5, 7, 5], [6, 6, 6, 1, 1, 1]]

  output = sequences_to_matrix(seqs, 10)
  print(output)

  output = pad_sequences(seqs, max_len=3)
  print(output)
