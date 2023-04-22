import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torch.utils.data
import torch.nn.utils.rnn as rnn_utils

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import time
import pickle
import pandas as pd


# embeddingModel_________________________________________________________________________________________________________________
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()


# treeLSTM_________________________________________________________________________________________________________________
class TreeLSTM(torch.nn.Module):
    '''PyTorch TreeLSTM model that implements efficient batching.
    '''

    def __init__(self, in_features, out_features, n_layer, dropout):
        '''TreeLSTM class initializer

        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layer = n_layer
        self.dropout = nn.Dropout(p=dropout)
        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.ModuleList(
            [torch.nn.Linear(self.in_features, 3 * self.out_features) for i in range(self.n_layer)])
        self.U_iou = torch.nn.ModuleList(
            [torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False) for i in range(self.n_layer)])
        self.W_f = torch.nn.ModuleList(
            [torch.nn.Linear(self.in_features, self.out_features) for i in range(self.n_layer)])
        self.U_f = torch.nn.ModuleList(
            [torch.nn.Linear(self.out_features, self.out_features, bias=False) for i in range(self.n_layer)])

    def forward(self, features, node_order, adjacency_list, edge_order, treesizes):
        '''Run TreeLSTM model on a tree data structure with node features

        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        '''

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]  # 一棵树中的节点数目

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = torch.device('cuda:0')  # 指定使用相同的cuda

        # h and c states for every node in the batch
        h = torch.zeros(self.n_layer, batch_size, self.out_features, device=device)  # h和c全为0
        c = torch.zeros(self.n_layer, batch_size, self.out_features, device=device)
        f_h = torch.zeros(self.n_layer, len(treesizes), self.out_features).to(device='cuda:0')
        f_c = torch.zeros(self.n_layer, len(treesizes), self.out_features).to(device='cuda:0')
        # f_h = 0
        # f_c = 0
        # populate the h and c states respecting computation order
        for i in range(self.n_layer):
            if i == 0:
                for n in range(int(node_order.max()) + 1):  # node_order.max（）寻找node_order中最大的元素
                    self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order, i)
                f_h[i, :, :], f_c[i, :, :] = self.findStartTree(node_order, h, c, treesizes, i)
            else:
                for n in range(int(node_order.max()) + 1):  # node_order.max（）寻找node_order中最大的元素
                    self._run_lstm(n, h, c, h[i - 1, :, :], node_order, adjacency_list, edge_order, i)
                f_h[i, :, :], f_c[i, :, :] = self.findStartTree(node_order, h, c, treesizes, i)

        return f_h, f_c

    def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order, num_layer_now):
        '''Helper function to evaluate all tree nodes currently able to be evaluated.
        '''
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration  # 判断node_order中是否含有与iteration相等的元素
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration  # 判断edge_order中是否含有与iteration相等的元素
        # x is a tensor of size n x F
        #         print(node_mask.shape)
        #         print(features.shape)
        x = features[node_mask, :]
        x = self.dropout(x)
        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou[num_layer_now](x)
        else:
            # adjacency_list is a tensor of size e x 2
            #             print(adjacency_list.shape)
            #             print(edge_mask.shape)
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0].type(torch.int64)
            child_indexes = adjacency_list[:, 1].type(torch.int64)

            # child_h and child_c are tensors of size e x 1
            # print(child_indexes)
            # print(num_layer_now)
            child_h = h[num_layer_now, child_indexes, :]
            child_c = c[num_layer_now, child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            h_sum = torch.stack(parent_list)
            iou = self.W_iou[num_layer_now](x) + self.U_iou[num_layer_now](h_sum)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        # print("i:",i.shape)
        # print("o:",o.shape)
        # print("u:",u.shape)
        i = torch.sigmoid(i)  # 输入门
        o = torch.sigmoid(o)  # 输出门
        u = torch.tanh(u)  # Ctilda~

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[num_layer_now, node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f = self.W_f[num_layer_now](features[parent_indexes, :]) + self.U_f[num_layer_now](child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)
            c[num_layer_now, node_mask, :] = i * u + c_sum

        h[num_layer_now, node_mask, :] = o * torch.tanh(c[num_layer_now, node_mask])

    def findStartTree(self, node_order, h, c, treesizes, num_layer_now):
        # print(h.requires_grad)
        # print(c.requires_grad)
        idx_list = list()
        count = 0
        for i in range(len(treesizes)):
            if i == 0:
                idx = torch.argmax(node_order[:treesizes[i]])

            else:
                idx = torch.argmax(node_order[count:count + int(treesizes[i])])
            count += int(treesizes[i])
            idx_list.append(idx)

        finall_c = torch.zeros(1, len(idx_list), self.out_features)
        finall_h = torch.zeros(1, len(idx_list), self.out_features)
        for k in range(len(idx_list)):
            finall_h[0, k, :] = h[num_layer_now, idx_list[k], :]
            finall_c[0, k, :] = c[num_layer_now, idx_list[k], :]

        return finall_h, finall_c


# token_LSTM_____________________________________________________________________________________________________________
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
        with open(r'modelData-512-lr=0.1/idx2word.pkl', 'rb') as f2:
            self.Tidx2word = pickle.load(f2)
        with open(r'modelData-512-lr=0.1/word2idx.pkl', 'rb') as f3:
            self.Tword2idx = pickle.load(f3)
        MAX_VOCAB_SIZE = 10000
        EMBEDDING_SIZE = 512
        model1 = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        model1.load_state_dict(torch.load(r'modelData-512-lr=0.1/embedding-512.th'))
        self.Tembedding_weights = model1.input_embedding()
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, sen_len):
        # src = [src sent len, batch size] 这句话的长度和batch大小
        embedded = torch.zeros(src.shape[0], src.shape[1], self.emb_dim).to(device)
        for i in range(len(src)):
            for k in range(len(src[i])):
                embedded[i, k, :] = torch.Tensor(self.embedding_Tword(int(src[i][k])))

        outputs, (hidden, cell) = self.rnn(embedded)
        ot = None
        for i in range(outputs.shape[1]):
            if i == 0:
                ot = outputs[sen_len[i] - 1, i, :]
            else:
                ot = torch.cat((ot, outputs[sen_len[i] - 1, i, :]), 0)

        # hidden, cell: [n layers* n directions, batch size, hid dim]

        return ot

    def embedding_Tword(self, word):
        embedding = self.Tembedding_weights[word]
        return embedding


# seq_LSTM_____________________________________________________________________________________________________________________
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
        with open(r'modelData-512-lr=0.1/idx2word.pkl', 'rb') as f:
            self.idx2word = pickle.load(f)
        with open(r'modelData-512-lr=0.1/word2idx.pkl', 'rb') as f1:
            self.word2idx = pickle.load(f1)
        MAX_VOCAB_SIZE = 10000
        EMBEDDING_SIZE = 512
        model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        model.load_state_dict(torch.load(r'modelData-512-lr=0.1/embedding-512.th'))
        self.embedding_weights = model.input_embedding()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.w_omega = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        self.u_omega = nn.Parameter(torch.Tensor(hid_dim, 1))
        self.decoder = nn.Linear(2 * hid_dim, 2)

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, src, token_lenth):
        # src = [src sent len, batch size] 这句话的长度和batch大小
        idx_list = list()
        count = 0
        for i in range(len(token_lenth)):
            if i == 0:
                idx = src[:token_lenth[i]]

            else:
                idx = src[count:count + int(token_lenth[i])]
            count += int(token_lenth[i])
            idx = idx.to(torch.device('cpu'))
            idx_list.append(torch.Tensor(idx))
        pad_value = torch.zeros(self.hid_dim)
        embedded = self.pad_sequence1(idx_list, pad_value)
        embedded = embedded.permute(1, 0, 2).to(torch.device('cuda:0'))

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
        att_score1 = torch.zeros(att_score.shape[0], att_score.shape[1], att_score.shape[2])
        for i in range(len(token_lenth)):
            att_score1[i, :token_lenth[i], :] = att_score[i, :token_lenth[i], :]
        att_score1 = att_score1.to(torch.device('cuda:0'))
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

    def pad_sequence1(self, sequences, padding_value):

        max_len = max([s.size(0) for s in sequences])

        out_tensor = torch.zeros(len(sequences), max_len, sequences[0].shape[1])
        for i, tensor in enumerate(sequences):
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :, ...] = padding_value

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :length, ...] = tensor

        return out_tensor


# FullModel_________________________________________________________________________________________________________________
class Fullmodel(nn.Module):
    def __init__(self, encoder, decode, decoder, device, hidden_size, output_size, num_layer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.decode = decode
        self.relu = torch.nn.ReLU()
        self.liner2 = nn.Linear(output_size, 1)
        self.num_layer = num_layer - 1

    def forward(self, features, node_order, adjacency_list, edge_order, treesizes, trg, token_lenth, sen_len):
        # batch_size = trg.shape[1]
        # max_len = trg.shape[0]
        # trg_vocab_size = self.decoder.output_dim
        # outputs = torch.zeros(max_len, batch_size,
        #                       trg_vocab_size).to(self.device)
        output = self.decode.forward(trg, sen_len)
        outpute, aa = self.decoder.forward(output, token_lenth)
        # print(type())
        # print(type(hidden))
        # print(type(output))
        y = self.liner2(outpute)
        outputs11 = torch.sigmoid(y)
        return outpute, outputs11, aa


# Loss____________________________________________________________________________________________________________________
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        device = (torch.device('cuda:0'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)

        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
            num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


# pro_dataset____________________________________________________________________________________________________________

class pro_DataSet(torch.utils.data.Dataset):
    def __init__(self, filename, limit_num, pro_num):
        # 词向量embedding(文件路径要改)
        # token序列嵌入词向量
        with open(r'modelData-512-lr=0.1/idx2word.pkl', 'rb') as f:
            self.idx2word = pickle.load(f)
        with open(r'modelData-512-lr=0.1/word2idx.pkl', 'rb') as f1:
            self.word2idx = pickle.load(f1)
        MAX_VOCAB_SIZE = 10000
        EMBEDDING_SIZE = 512
        model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        model.load_state_dict(torch.load(r'modelData-512-lr=0.1/embedding-512.th'))
        self.embedding_weights = model.input_embedding()
        # tree嵌入词向量
        with open(r'modelData-tree-512-lr0.1/idx2word.pkl', 'rb') as f2:
            self.Tidx2word = pickle.load(f2)
        with open(r'modelData-tree-512-lr0.1/word2idx.pkl', 'rb') as f3:
            self.Tword2idx = pickle.load(f3)
        # MAX_VOCAB_SIZE = 10000
        # EMBEDDING_SIZE = 512
        model1 = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        model1.load_state_dict(torch.load(r'modelData-tree-512-lr0.1/embedding-512.th'))
        self.Tembedding_weights = model1.input_embedding()
        # 读入数据
        de = pd.read_json(filename, lines=True)
        # 给每个token嵌入词向量
        data_list = list()
        pad_value = self.idx_word("<pad>")

        for i in range(len(de)):
            if i % 2 == 0:
                p_num = None
                p_name = None
                for pr_nu in de.loc[i, ["pro_num"]].values:
                    p_num = pr_nu
                for pr_na in de.loc[i, ["pro_name"]].values:
                    p_name = pr_na
                this_name = str(p_name) + str(p_num)
                if this_name in pro_num:
                    print(this_name)

                    continue
                data_dict = {
                    "token_seq": None,
                    "tokenseq_len": None,
                    "token_tree": None,
                    "adjacency": None,
                    "node_order": None,
                    "edge_order": None,
                    "tree_sizes": None,
                    "feature_label": None,
                    "sen_labels": None
                }
                token_seq, tokenseq_len = self.getNewtoken(limit_num, i, de)
                token_seq1, tokenseq_len1 = self.getNewtoken(limit_num, i + 1, de)
                token_seq.extend(token_seq1)
                tokenseq_len.extend(tokenseq_len1)

                #
                sequences = list()
                len_sequences = list()
                for seq in token_seq:
                    sequence = list()
                    for tokens in seq:
                        sequence.append(self.idx_word(tokens))
                    sequences.append(torch.Tensor(sequence))
                    len_sequences.append(len(sequence))
                batched_codeSquences = rnn_utils.pad_sequence(sequences, batch_first=True,
                                                              padding_value=pad_value)
                data_dict["token_seq"] = batched_codeSquences
                data_dict["tokenseq_len"] = tokenseq_len
                batch_list = list()
                sl = list()
                se = list()
                for k in range(2):
                    token_tree = self.getNewTree(limit_num, i + k, de)
                    tree_value = list()
                    for seq in token_tree:
                        sens_calue = list()
                        for seqs in seq:
                            node_values = list()
                            for tokens in seqs:
                                node_values.append(self.embedding_Tword(tokens))

                            sens_calue.append(node_values)
                        tree_value.append(sens_calue)
                    adjacency_lists, node_orders, edge_orders, seq_labels, sen_labels = self.getOtherThree(limit_num,
                                                                                                           i + k, de)
                    sl.extend(seq_labels)
                    se.extend(sen_labels)

                    for limit in range(limit_num):
                        batch_dict = {
                            "features": None,
                            "node_order": None,
                            "edge_order": None,
                            "adjacency_list": None,
                        }
                        batch_dict["features"] = tree_value[limit]
                        batch_dict["node_order"] = node_orders[limit]
                        batch_dict["edge_order"] = edge_orders[limit]
                        batch_dict["adjacency_list"] = adjacency_lists[limit]
                        batch_list.append(batch_dict)
                batched_features, batched_node_order, batched_edge_order, batched_adjacency_list, tree_sizes = self.batch_tree_input(
                    batch_list)
                se = self.pad_sequence2(se, -1)
                data_dict["token_tree"] = batched_features
                data_dict["adjacency"] = batched_adjacency_list
                data_dict["edge_order"] = batched_edge_order
                data_dict["node_order"] = batched_node_order
                data_dict["feature_label"] = sl
                data_dict["sen_labels"] = se
                data_dict["tree_sizes"] = tree_sizes
                data_dict["len_sen"] = len_sequences
                data_list.append(data_dict)
        self.data = data_list

    def __getitem__(self, idx):  # if the index is idx, what will be the data?
        return self.data[idx]["token_seq"], self.data[idx]["token_tree"], self.data[idx]["node_order"], self.data[idx][
            "adjacency"], self.data[idx]["edge_order"], self.data[idx]["tree_sizes"], self.data[idx]["feature_label"], \
               self.data[idx]["tokenseq_len"], self.data[idx]["sen_labels"], self.data[idx]["len_sen"]

    def __len__(self):  # What is the length of the dataset
        return len(self.data)

    # def embedding_word(self,word):
    #     index = self.word2idx[word]
    #     embedding = self.embedding_weights[index]
    #     return embedding
    def embedding_Tword(self, word):
        try:
            index = self.Tword2idx[word]
        except:
            index = self.Tword2idx['<UNK>']
        embedding = self.Tembedding_weights[index]
        return embedding

    def idx_word(self, word):
        try:
            index = self.word2idx[word]
        except:
            index = self.word2idx['<UNK>']
        return index

    def batch_tree_input(self, batch):
        '''Combines a batch of tree dictionaries into a single batched dictionary for use by the TreeLSTM model.

        batch - list of dicts with keys ('features', 'node_order', 'edge_order', 'adjacency_list')
        returns a dict with keys ('features', 'node_order', 'edge_order', 'adjacency_list', 'tree_sizes')
        '''
        tree_sizes = list()
        batched_features = list()
        batched_node_order1 = list()
        batched_edge_order1 = list()
        batched_adjacency_list1 = list()
        # print("batch_len: ",len(batch))
        # for b in batch:
        #     tree_sizes.append(len(b['features']))
        #     batched_features.append(torch.Tensor(b['features']))

        for b in batch:
            for c in b['features']:
                tree_sizes.append(len(c))
                batched_features.append(torch.Tensor(c))

        for b in batch:
            for c in b['node_order']:
                batched_node_order1.append(torch.Tensor(c))

        for b in batch:
            for c in b['edge_order']:
                batched_edge_order1.append(torch.Tensor(c))
        for b in batch:
            for c in b['adjacency_list']:
                batched_adjacency_list1.append(torch.Tensor(c))

        batched_features = torch.cat([b for b in batched_features])
        batched_node_order = torch.cat([b for b in batched_node_order1])
        batched_edge_order = torch.cat([b for b in batched_edge_order1])
        batched_adjacency_list = []
        offset = 0
        for n, b in zip(tree_sizes, batched_adjacency_list1):
            batched_adjacency_list.append(b + offset)
            offset += n
        batched_adjacency_list = torch.cat(batched_adjacency_list)

        # tree_sizes = [b['features'].shape[0] for b in batch]
        # batched_features = torch.cat([b['features'] for b in batch])
        #
        # batched_node_order = torch.cat([b['node_order'] for b in batch])
        # batched_edge_order = torch.cat([b['edge_order'] for b in batch])
        #
        # batched_adjacency_list = []
        # offset = 0
        # for n, b in zip(tree_sizes, batch):
        #     batched_adjacency_list.append(b['adjacency_list'] + offset)
        #     offset += n
        # batched_adjacency_list = torch.cat(batched_adjacency_list)
        #
        return batched_features, batched_node_order, batched_edge_order, batched_adjacency_list, tree_sizes

    def getNewtoken(self, limit_num, num, de):
        all_tokenDIC = list()
        limit_token = {
            "para": 55,
            "var": 818,
            "Rtype": 819,
            "func": 915,
            "method": 36,
            "memberRef": 914,
            "DecimalInteger": 743,
            "OtherClass": 395,
            "String": 749,
            "DecimalFloatingPoint": 155,
            "HexInteger": 352,
            "class": 12,
            "HexFloatingPoint": 2,
            "Annotation": 1,
            "BinaryInteger": 17,
        }
        flag = 0
        for limit in range(limit_num):
            tokenDIC = dict()
            for tokenDICTS in de.loc[num, ["tokenDIC"]].values:
                tokenDIC = copy.deepcopy(tokenDICTS)

                for k_tokenDIC in tokenDIC:
                    num_str = str()
                    str_str = str()

                    for s in tokenDIC[k_tokenDIC]:
                        if s.isdigit():
                            flag = 1
                            break
                    if flag == 0:
                        continue
                    else:
                        flag = 0
                    for s in tokenDIC[k_tokenDIC]:
                        if s.isdigit():
                            num_str += s

                        else:
                            str_str += s
                    if str_str == "para" or str_str == "var" or str_str == "Rtype" or str_str == "func" or str_str == "method" or str_str == "memberRef" or str_str == "DecimalInteger" or str_str == "OtherClass" or str_str == "String" or str_str == "DecimalFloatingPoint" or str_str == "HexInteger" or str_str == "class" or str_str == "BinaryInteger" or str_str == "Annotation" or str_str == "HexFloatingPoint":
                        if int(num_str) <= limit_token[str_str]:
                            if int(num_str) + limit <= limit_token[str_str]:
                                tokenDIC[k_tokenDIC] = str_str + str(int(num_str) + limit)
                            else:
                                tokenDIC[k_tokenDIC] = str_str + str((int(num_str) + limit) % limit_token[str_str])
                        else:
                            pass
                tokenDIC = tokenDIC
            all_tokenDIC.append(tokenDIC)
        all_tokenList = list()
        all_seqlen = list()
        for limit in range(limit_num):
            for token_seqs in de.loc[num, ["codeSquence"]].values:
                all_seqlen.append(len(token_seqs))
                for sens in token_seqs:
                    token_seq = list()
                    for token in sens:
                        try:
                            token_seq.append(all_tokenDIC[limit][token])
                        except KeyError:
                            token_seq.append(token)
                    all_tokenList.append(token_seq)

        return all_tokenList, all_seqlen

    def getNewTree(self, limit_num, num, de):
        flag = 0
        all_treeDIC = list()

        limit_tree = {
            "para": 55,
            "var": 818,
            "Rtype": 819,
            "func": 915,
            "method": 36,
            "memberRef": 914,
            "DecimalInteger": 743,
            "OtherClass": 256,
            "String": 749,
            "DecimalFloatingPoint": 155,
            "HexInteger": 352,
            "class": 12,
            "BinaryInteger": 17,
        }
        for limit in range(limit_num):
            treeDIC = dict()
            for treeDICTS in de.loc[num, ["treeDICT"]].values:
                treeDICTS1 = copy.deepcopy(treeDICTS)
                for k_tokenDIC in treeDICTS1:
                    num_str = str()
                    str_str = str()

                    for s in treeDICTS1[k_tokenDIC]:
                        if s.isdigit():
                            flag = 1
                            break
                    if flag == 0:
                        continue
                    else:
                        flag = 0
                    for s in treeDICTS1[k_tokenDIC]:
                        if s.isdigit():
                            num_str += s

                        else:
                            str_str += s
                    if str_str == "para" or str_str == "var" or str_str == "Rtype" or str_str == "func" or str_str == "method" or str_str == "memberRef" or str_str == "DecimalInteger" or str_str == "OtherClass" or str_str == "String" or str_str == "DecimalFloatingPoint" or str_str == "HexInteger" or str_str == "class" or str_str == "BinaryInteger":
                        if int(num_str) <= limit_tree[str_str]:
                            if int(num_str) + limit <= limit_tree[str_str]:
                                treeDICTS1[k_tokenDIC] = str_str + str(int(num_str) + limit)
                            else:
                                treeDICTS1[k_tokenDIC] = str_str + str((int(num_str) + limit) % limit_tree[str_str])
                        else:
                            pass
                treeDIC = treeDICTS1
            all_treeDIC.append(treeDIC)

        all_treeList1 = list()
        for limit in range(limit_num):
            all_treeList = list()
            for token_seqs in de.loc[num, ["node_value"]].values:
                for sens in token_seqs:
                    token_seq = list()
                    for token in sens:
                        try:
                            token_seq.append(all_treeDIC[limit][token])
                        except KeyError:
                            token_seq.append(token)
                    all_treeList.append(token_seq)
            all_treeList1.append(all_treeList)
        return all_treeList1

    # data_dict = {
    #     "treeDICT": None,
    #     "tokenDIC": None,
    #     "codeSquence": list(),
    #     "node_value": list(),
    #     "adjacency_list": list(),
    #     "node_order": list(),
    #     "edge_order": list(),
    #     "seq_label": None,
    #     "pro_num": None,
    #     "pro_name": None
    # }
    def pad_sequence2(self, sequences, padding_value):
        max_len = max([s.size(0) for s in sequences])
        out_tensor = torch.zeros(len(sequences), max_len)
        for i, tensor in enumerate(sequences):
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :] = padding_value

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :length] = tensor

        return out_tensor

    def getOtherThree(self, limit_num, num, de):
        node_orders = list()
        edge_orders = list()
        adjacency_lists = list()
        seq_labels = list()
        sen_labels = list()
        for limit in range(limit_num):
            for node_order in de.loc[num, ["node_order"]].values:
                node_orders.append(node_order)
            for edge_order in de.loc[num, ["edge_order"]].values:
                edge_orders.append(edge_order)
            for adjacency in de.loc[num, ["adjacency_list"]].values:
                adjacency_lists.append(adjacency)
            for seq_label in de.loc[num, ["seq_label"]].values:
                seq_labels.append(seq_label)
            for seq_label in de.loc[num, ["sen_label"]].values:
                sen_labels.append(torch.Tensor(seq_label))

        # sen_labels = self.pad_sequence2(sen_labels, -1)
        return adjacency_lists, node_orders, edge_orders, seq_labels, sen_labels


# main_____________________________________________________________________________________________________________________
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, criterion1, criterion2, clip):
    model.train()

    epoch_loss = 0

    for i, (trg, features, node_order, adjacency_list, edge_order, treesizes, feature_label, tokenseq_len, sen_label,
            sen_len) in enumerate(iterator):
        feature_label = torch.Tensor(feature_label)
        optimizer.zero_grad()
        trg = trg.reshape(trg.shape[1], trg.shape[2])
        trg = trg.permute(1, 0)
        features = features.reshape(features.shape[1], features.shape[2])
        node_order = node_order.reshape(node_order.shape[1])
        adjacency_list = adjacency_list.reshape(adjacency_list.shape[1], adjacency_list.shape[2])
        edge_order = edge_order.reshape(edge_order.shape[1])

        trg = trg.to(device)
        node_order = node_order.to(device)
        adjacency_list = adjacency_list.to(device)
        edge_order = edge_order.to(device)
        #         treesizes = treesizes.to(device)
        features = features.to(device)
        output, output1, aa = model.forward(features, node_order, adjacency_list, edge_order, treesizes, trg,
                                            tokenseq_len, sen_len)

        feature_label = feature_label.to(device)

        loss = criterion.forward(output, feature_label)

        feature_label = feature_label.unsqueeze(1)

        loss1 = criterion1.forward(output1, feature_label)
        cc = aa.permute(2, 0, 1)
        sen_label = sen_label.to(device)
        mask = sen_label.ge(0)
        sen_label = torch.masked_select(sen_label, mask)

        cc = torch.masked_select(cc, mask)

        loss3 = criterion2.forward(cc, sen_label)
        loss2 = loss1 + loss * 10 + loss3 * 100
        loss2.requires_grad_(True)
        loss2.backward()

        # gradient clipping 防止梯度爆炸问题
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss2.item()
    return epoch_loss / len(iterator)


SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
BATCH_SIZE = 1
limit_num = 4
# 参数设置
INPUT_DIM = 512  # 词表的词嵌入维度
# OUTPUT_DIM = len(word2idx)
ENC_EMB_DIM = 512  # 模型输出的维度
DEC_EMB_DIM = 512
HID_DIM = 512
HID_SIZE = 1024
N_LAYERS = 2
OUTPUTSIZE = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0
enc = TreeLSTM(INPUT_DIM, ENC_EMB_DIM, N_LAYERS, DEC_DROPOUT)
dec = Encoder1(DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
deco = Encoder2(DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
device = torch.device('cuda:0')
model = Fullmodel(enc, dec, deco, device, HID_SIZE, OUTPUTSIZE, N_LAYERS).to(device)
b = np.load('a0.npy')
b = b.tolist()
dataset1 = pro_DataSet(r"data/data_new1.json", limit_num, b)  # create the dataset
dataloader = torch.utils.data.DataLoader(dataset=dataset1, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
N_EPOCHS = 10
CLIP = 1

# best_valid_loss = float('inf')
criterion = SupConLoss()
criterion1 = nn.BCELoss()
criterion2 = nn.BCELoss()

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, dataloader, optimizer, criterion, criterion1, criterion2, CLIP)
    # valid_loss = evaluate(model, dataloader1, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    torch.save(model, 'seq-0-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.6f}')