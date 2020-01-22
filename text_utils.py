import os
import json
import collections
import tensorflow as tf
import tqdm
import numpy as np
import pprint

from gensim.models.keyedvectors import KeyedVectors

from nltk.tokenize import TweetTokenizer
global _TOKENIZER
_TOKENIZER = TweetTokenizer()


def preprocess_caption(cap_in):
    return ' '.join(_TOKENIZER.tokenize(cap_in)).lower()


def get_vocab(data, min_count=5, cached=None):
    if cached is None or not os.path.exists(cached):
        voc_counter = collections.Counter()
        for c in tqdm.tqdm(data):
            voc_counter.update(preprocess_caption(c).split())
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = len(word2idx)
        for v, c in sorted(voc_counter.items(), key=lambda x: x[1], reverse=True):
            if c < min_count:
                break
            word2idx[v] = idx
            idx += 1
        if cached is not None:
            with open(cached, 'w') as f:
                f.write(json.dumps(word2idx))
    else:
        with open(cached) as f:
            word2idx = json.loads(f.read())
    return word2idx


def get_word2vec_matrix(vocab, cache_file, word2vec_binary):
    if cache_file is None and word2vec_binary is None:
        return None
    if cache_file is None or not os.path.exists(cache_file):
        print('Loading word2vec binary...')
        word2vec = KeyedVectors.load_word2vec_format(word2vec_binary, binary=True)
        word2vec_cachable = {}
        for w, idx in vocab.items():
            if w in word2vec:
                word2vec_cachable[w] = list([float(x) for x in word2vec[w]])
        if cache_file is not None:
            with open(cache_file, 'w') as f:
                f.write(json.dumps(word2vec_cachable))
    else:
        with open(cache_file) as f:
            word2vec_cachable = json.loads(f.read())
    word2vec = {w:np.array(v) for w, v in word2vec_cachable.items()}
    m_matrix = np.random.uniform(-.2, .2, size=(len(vocab), 300))
    for w, idx in vocab.items():
        if w in word2vec:
            m_matrix[idx, :] = word2vec[w]
    return m_matrix


def text_to_matrix(captions, vocab, max_len=15, padding='pre'):
    seqs = []
    for c in captions:
        tokens = preprocess_caption(c).split()

        # for reasons I dont understand, the new version of CUDNN
        # doesn't play nice with padding, etc.  After painstakingly
        # narrowing down why this happens, CUDNN errors when:
        
        # 1) you're using RNN
        # 2) your batch consists of fully-padded sequences and non-padded sequences only

        # I filed a tensorflow issue:
        # see https://github.com/tensorflow/tensorflow/issues/36139

        # To solve this, for now, I'm just adding a padding char to all sequences.
        # This padding token can be removed in the future when this issue is fixed.
        
        idxs = [vocab['<PAD>']] + [vocab[v] if v in vocab else vocab['<UNK>'] for v in tokens]
        seqs.append(idxs)
    m_mat = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_len+1,
                                                          padding=padding, truncating='post',
                                                          value=0)
    return m_mat
