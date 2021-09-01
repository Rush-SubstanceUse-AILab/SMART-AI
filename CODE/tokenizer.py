#!/usr/bin/env python3

import collections

class Tokenizer:
  """Tokenization and vectorization"""

  def __init__(
   self,
   n_words,
   lower=False,
   padding_token='PAD',
   unk_token='UNK',
   cls_token='CLS',
   token=True):
    """Construction deconstruction"""

    self.stoi = {}
    self.itos = {}
    self.counts = collections.Counter()

    # first three tokens are reserved
    self.n_words = None if n_words is None else n_words-3
    self.lower = lower
    self.unk = unk_token
    self.cls = cls_token

    if token is True:
        self.stoi[padding_token] = 0
        self.itos[0] = padding_token
        self.stoi[unk_token] = 1
        self.itos[1] = unk_token
        self.stoi[cls_token] = 2
        self.itos[2] = cls_token
    else:
        #self.stoi[unk_token] = 0
        pass

  def fit_on_texts(self, texts, tok=True):
    """Fit on a list of document strings"""

    for text in texts:
      tokens = text.split( )
      self.counts.update(tokens)

    if tok is True:
      index = 3 # 0, 1, 2 already taken
      for token, cnt in self.counts.most_common(self.n_words):
        if cnt > 0:
          self.stoi[token] = index
          self.itos[index] = token
          index += 1

    else:
      index = 0
      for token, cnt in self.counts.most_common(self.n_words):
        if cnt > 0:
          self.stoi[token] = index
          self.itos[index] = token
          index += 1


  def texts_to_seqs(self, texts, add_cls_token=False):
    """List of strings to list of int sequences"""

    sequences = []
    #print(self.stoi)
    for text in texts:

      sequence = []

      if add_cls_token:
        sequence.append(self.stoi[self.cls])

      for token in text.split( ):
        #print(token)
        if str(token) in self.stoi:
          sequence.append(self.stoi[str(token)])
        else:
          #print(self.unk)
          #print(token)
          #sequence.append(self.stoi[self.oov])
          sequence.append(self.stoi[self.unk])

      sequences.append(sequence)

    return sequences

  def texts_as_sets_to_seqs(self, texts, add_cls_token=False):
    """Same as texts_to_sequences but treat texts as sets"""

    sequences = []
    for text in texts:

      sequence = []
      if add_cls_token:
        sequence.append(self.stoi[self.cls])

      for token in set(text.split()):
        if token in self.stoi:
          sequence.append(self.stoi[token])
        else:
          sequence.append(self.stoi[self.unk])

      sequences.append(sequence)

    return sequences

if __name__ == "__main__":

  sents = ['it is happening again',
           'the owls are not what they seem',
           'again and again',
           'the owls are happening']

  tokenizer = Tokenizer(n_words=6)

  tokenizer.fit_on_texts(sents)
  print('counts:', tokenizer.counts)
  print('stoi:', tokenizer.stoi)
  print('itos:', tokenizer.itos)

  seqs = tokenizer.texts_to_seqs(sents, add_cls_token=True)
  print(seqs)
