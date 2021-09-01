#!/usr/bin/env python3

import sys
import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.autograd import Variable
import random

torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(2020)


class BagOfEmbeddings(nn.Module):

  def __init__(
    self,
    input_vocab_size,
    output_vocab_size,
    embed_dim,
    hidden_units,
    dropout_rate,
    out_size,
    stride,
    kernel,
    save_config=True):
    """Constructor"""

    super(BagOfEmbeddings, self).__init__()

    self.embed = nn.Embedding(
      num_embeddings=input_vocab_size,
      embedding_dim=embed_dim)

    #self.hidden = nn.Linear(
      #in_features=embed_dim,
      #out_features=hidden_units)

    self.activation = nn.ReLU()

    self.dropout = nn.Dropout(dropout_rate)
    
    #self.kernel_1 = [3]

    self.embed_dim = embed_dim
    self.kernel_1 = [kernel]
    self.out_size = out_size

    self.convs = nn.ModuleList([nn.Conv2d(1, self.out_size, (K, embed_dim)) for K in self.kernel_1])
    
    #self.pool_1 = nn.MaxPool1d(self.kernel_1, int(self.out_size))

    self.hidden_layer = nn.Linear(len(self.kernel_1) * self.out_size, hidden_units)
    self.fc1 = nn.Linear(hidden_units, output_vocab_size)

    if save_config:
      config = {
        'input_vocab_size': input_vocab_size,
        'output_vocab_size': output_vocab_size,
        'embed_dim': embed_dim,
        'hidden_units': hidden_units,
        'dropout_rate': dropout_rate,
        'out_size': out_size,
        'stride':stride,
        'kernel':kernel}
      pickle_file = open(config_path, 'wb')
      pickle.dump(config, pickle_file)


  def in_features_fc(self):
    out_conv_1=((self.embed_dim - 1 * (self.kernel_1 - 1) -1) / self.stride) + 1
    out_conv_1 = math.floor(out_conv_1)
    out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) -1) / self.stride) + 1
    out_pool_1 = math.floor(out_pool_1)
    return out_pool_1 * self.out_size

  def forward(self, texts, return_hidden=False):
    """Optionally return hidden layer activations"""

    x = self.embed(texts)
    x = x.unsqueeze(1)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    x = torch.cat(x, 1)
    x = self.dropout(x)
    x = self.hidden_layer(x)
    out = self.fc1(x)

    if return_hidden:
      return features
    else:
      return out



if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  
  #mainRan()
