import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from EmbeddingModel import EmbeddingModel
import torch.nn.utils.rnn as rnn_utils
import random
import math
import time
class Encoder2(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        '''

        :param input_dim: 输入源词库的大小
        :param emb_dim:  输入单词Embedding的维度
        :param hid_dim: 隐层的维度
        :param n_layers: 几个隐层
        :param dropout:  dropout参数 0.5
        '''

        super().__init__()
        with open(r'E:\pycharmworkSpace\tree_seq2seq\modelData-512-lr=0.1\idx2word.pkl', 'rb') as f:
            self.idx2word = pickle.load(f)
        with open(r'E:\pycharmworkSpace\tree_seq2seq\modelData-512-lr=0.1\word2idx.pkl', 'rb') as f1:
            self.word2idx = pickle.load(f1)
        MAX_VOCAB_SIZE = 10000
        EMBEDDING_SIZE = 512
        model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        model.load_state_dict(torch.load(r'E:\pycharmworkSpace\tree_seq2seq\modelData-512-lr=0.1\embedding-512.th'))
        self.embedding_weights = model.input_embedding()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers = n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.w_omega = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        self.u_omega = nn.Parameter(torch.Tensor(hid_dim, 1))
        self.decoder = nn.Linear(2 * hid_dim, 2)

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def forward(self, src,token_lenth):
        # src = [src sent len, batch size] 这句话的长度和batch大小
        idx_list = list()
        count = 0
        for i in range(len(token_lenth)):
            if i == 0:
                idx = src[:token_lenth[i]]

            else:
                idx = src[count:count + int(token_lenth[i])]
            count += int(token_lenth[i])
            idx_list.append(torch.Tensor(idx))
        pad_value = torch.zeros(self.hid_dim)
        embedded = self.pad_sequence1(idx_list,pad_value)
        embedded = embedded.permute(1,0,2)

        # embedded = [src sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        x = outputs.permute(1, 0, 2)
        # x形状是(batch_size, seq_len, 2 * num_hiddens)

        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        aa = torch.sigmoid(att)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        att_score1 = torch.zeros(att_score.shape[0],att_score.shape[1],att_score.shape[2])
        for i in range(len(token_lenth)):
            att_score1[i,:token_lenth[i],:] = att_score[i,:token_lenth[i],:]
        scored_x = x * att_score1
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(batch_size, 2 * num_hiddens)
        # outs = self.decoder(feat)
        # out形状是(batch_size, 2)


        # hid = hidden[self.n_layers-1, :, :]
        return feat, aa

    def find_embd(self, word):
        index = self.word2idx[word]
        embedding = self.embedding_weights[index]
        return embedding

    def pad_sequence1(self,sequences, padding_value):

        max_len = max([s.size(0) for s in sequences])


        out_tensor = torch.zeros(len(sequences),max_len,sequences[0].shape[1])
        for i, tensor in enumerate(sequences):
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :, ...] = padding_value

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :length, ...] = tensor


        return out_tensor