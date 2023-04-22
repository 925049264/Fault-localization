import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from EmbeddingModel import EmbeddingModel

import random
import math
import time
class Encoder1(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        '''

        :param input_dim: 输入源词库的大小
        :param emb_dim:  输入单词Embedding的维度
        :param hid_dim: 隐层的维度
        :param n_layers: 几个隐层
        :param dropout:  dropout参数 0.5
        '''

        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        with open(r'E:\pycharmworkSpace\tree_seq2seq\modelData-512-lr=0.1\idx2word.pkl', 'rb') as f2:
            self.Tidx2word = pickle.load(f2)
        with open(r'E:\pycharmworkSpace\tree_seq2seq\modelData-512-lr=0.1\word2idx.pkl', 'rb') as f3:
            self.Tword2idx = pickle.load(f3)
        MAX_VOCAB_SIZE = 10000
        EMBEDDING_SIZE = 512
        model1 = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        model1.load_state_dict(torch.load(r'E:\pycharmworkSpace\tree_seq2seq\modelData-512-lr=0.1\embedding-512.th'))
        self.Tembedding_weights = model1.input_embedding()
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers = n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src,sen_len):
        # src = [src sent len, batch size] 这句话的长度和batch大小
        embedded = torch.zeros(src.shape[0], src.shape[1], self.emb_dim)
        for i in range(len(src)):
            for k in range(len(src[i])):
                embedded[i,k,:] = torch.Tensor(self.embedding_Tword(int(src[i][k])))
        outputs, (hidden, cell) = self.rnn(embedded)
        ot = None
        for i in range(outputs.shape[1]):
            if i == 0:
                ot = outputs[sen_len[i]-1,i,:]
            else:
                ot = torch.cat((ot,outputs[sen_len[i]-1,i,:]),0)

        # hidden, cell: [n layers* n directions, batch size, hid dim]

        return ot
    def embedding_Tword(self,word):
        embedding = self.Tembedding_weights[word]
        return embedding