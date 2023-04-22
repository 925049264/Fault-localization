import copy
import os
import re

import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.utils.data
import torch.nn.utils.rnn as rnn_utils
from EmbeddingModel import EmbeddingModel
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torch.utils.data
import torch.nn.utils.rnn as rnn_utils


class TreeLSTM(torch.nn.Module):
    '''PyTorch TreeLSTM model that implements efficient batching.
    '''
    def __init__(self, in_features, out_features, n_layer):
        '''TreeLSTM class initializer

        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layer = n_layer
        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.ModuleList([torch.nn.Linear(self.in_features, 3 * self.out_features) for i in range(self.n_layer)])
        self.U_iou = torch.nn.ModuleList([torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False) for i in range(self.n_layer)])
        self.W_f = torch.nn.ModuleList([torch.nn.Linear(self.in_features, self.out_features) for i in range(self.n_layer)])
        self.U_f = torch.nn.ModuleList([torch.nn.Linear(self.out_features, self.out_features, bias=False) for i in range(self.n_layer)])

    def forward(self, features, node_order, adjacency_list, edge_order, treesizes):
        '''Run TreeLSTM model on a tree data structure with node features

        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        '''

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]#一棵树中的节点数目

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = torch.device('cpu')#指定使用相同的cuda

        # h and c states for every node in the batch
        h = torch.zeros(self.n_layer, batch_size, self.out_features, device=device)#h和c全为0
        c = torch.zeros(self.n_layer, batch_size, self.out_features, device=device)
        f_h = torch.zeros(self.n_layer, len(treesizes), self.out_features)
        f_c = torch.zeros(self.n_layer, len(treesizes), self.out_features)
        # f_h = 0
        # f_c = 0
        # populate the h and c states respecting computation order
        # print(int(node_order.max()) + 1)
        for i in range(self.n_layer):
            if i == 0:
                for n in range(int(node_order.max()) + 1):  # node_order.max（）寻找node_order中最大的元素
                    self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order,i)
                f_h[i, :, :], f_c[i, :, :] = self.findStartTree(node_order, h, c, treesizes,i)
            else:
                for n in range(int(node_order.max()) + 1):  # node_order.max（）寻找node_order中最大的元素
                    self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order,i)
                f_h[i, :, :], f_c[i, :, :] = self.findStartTree(node_order, h, c, treesizes,i)



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
        x = features[node_mask, :]

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou[num_layer_now](x)
        else:
            # adjacency_list is a tensor of size e x 2
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
    def findStartTree(self, node_order, h, c, treesizes,num_layer_now):
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
#token_LSTM_____________________________________________________________________________________________________________
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
#seq_LSTM_____________________________________________________________________________________________________________________
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
#FullModel_________________________________________________________________________________________________________________
class Fullmodel(nn.Module):
    def __init__(self, encoder,decode, decoder, device,hidden_size, output_size, num_layer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.decode = decode
        self.liner1 = nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.liner2 = nn.Linear(output_size, 1)
        self.num_layer = num_layer-1

    def forward(self, features, node_order, adjacency_list, edge_order, treesizes, trg,token_lenth,sen_len):
        # batch_size = trg.shape[1]
        # max_len = trg.shape[0]
        # trg_vocab_size = self.decoder.output_dim
        # outputs = torch.zeros(max_len, batch_size,
        #                       trg_vocab_size).to(self.device)
        hidden, cell = self.encoder.forward(features, node_order, adjacency_list, edge_order, treesizes)
        output = self.decode.forward(trg,sen_len)
        hid = hidden[self.num_layer,:,:]

        output1 = torch.cat([hid, output], 1)
        outputs = self.liner1(output1)
        outpute, aa = self.decoder.forward(outputs,token_lenth)
        # print(type())
        # print(type(hidden))
        # print(type(output))
        y = self.liner2(outpute)
        outputs11 = torch.sigmoid(y)
        return outpute,outputs11, aa
class pro_DataSet(torch.utils.data.Dataset):
    def __init__(self,filename,limit_num,pro_num):
        # 词向量embedding(文件路径要改)
        # token序列嵌入词向量
        with open(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\idx2word.pkl', 'rb') as f:
            self.idx2word = pickle.load(f)
        with open(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\word2idx.pkl', 'rb') as f1:
            self.word2idx = pickle.load(f1)
        MAX_VOCAB_SIZE = 10000
        EMBEDDING_SIZE = 512
        model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        model.load_state_dict(torch.load(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\embedding-512.th'))
        self.embedding_weights = model.input_embedding()
        #tree嵌入词向量
        with open(r'E:\pycharmworkSpace\StatementContrastive\identifier\tree_vec\idx2word.pkl', 'rb') as f2:
            self.Tidx2word = pickle.load(f2)
        with open(r'E:\pycharmworkSpace\StatementContrastive\identifier\tree_vec\word2idx.pkl', 'rb') as f3:
            self.Tword2idx = pickle.load(f3)
        model1 = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        model1.load_state_dict(torch.load(r'E:\pycharmworkSpace\StatementContrastive\identifier\tree_vec\embedding-512.th'))
        self.Tembedding_weights = model1.input_embedding()
        # 读入数据
        de = pd.read_json(filename, lines=True)
        # 给每个token嵌入词向量
        data_list = list()
        pad_value = self.idx_word("<pad>")

        for i in range(len(de)):
            if i % 2 == 1:
                p_num = None
                p_name =None
                for pr_nu in de.loc[i, ["pro_num"]].values:
                    p_num = pr_nu
                for pr_na in de.loc[i, ["pro_name"]].values:
                    p_name = pr_na
                this_name = str(p_name) + str(p_num)
                if this_name not in pro_num:
                    continue
                data_dict = {
                    "token_seq": None,
                    "tokenseq_len":None,
                    "token_tree": None,
                    "adjacency": None,
                    "node_order": None,
                    "edge_order": None,
                    "tree_sizes": None,
                    "feature_label": None,
                    "sen_labels": None,
                    "this_name": None
                }
                token_seq,tokenseq_len = self.getNewtoken(limit_num, i, de)

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
                for k in range(1):
                    token_tree = self.getNewTree(limit_num, i+k, de)
                    tree_value = list()
                    for seq in token_tree:
                        sens_calue = list()
                        for seqs in seq:
                            node_values = list()
                            for tokens in seqs:
                                node_values.append(self.embedding_Tword(tokens))

                            sens_calue.append(node_values)
                        tree_value.append(sens_calue)
                    adjacency_lists, node_orders, edge_orders, seq_labels,sen_labels = self.getOtherThree(limit_num, i+k,de)
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
                        # print(batch_dict["features"].shape)
                        batch_dict["node_order"] = node_orders[limit]
                        batch_dict["edge_order"] = edge_orders[limit]
                        batch_dict["adjacency_list"] = adjacency_lists[limit]
                        batch_list.append(batch_dict)
                batched_features, batched_node_order, batched_edge_order, batched_adjacency_list, tree_sizes = self.batch_tree_input(batch_list)
                se = self.pad_sequence2(se, -1)
                data_dict["token_tree"] = batched_features
                data_dict["adjacency"] = batched_adjacency_list
                data_dict["edge_order"] = batched_edge_order
                data_dict["node_order"] = batched_node_order
                data_dict["feature_label"] = sl
                data_dict["sen_labels"] = se
                # data_dict["program_num"] = pn
                data_dict["tree_sizes"] = tree_sizes
                data_dict["len_sen"] = len_sequences
                data_dict["this_name"] = this_name
                data_list.append(data_dict)
        self.data = data_list

    def __getitem__(self, idx):  # if the index is idx, what will be the data?
        return self.data[idx]["token_seq"],self.data[idx]["token_tree"],self.data[idx]["node_order"],self.data[idx]["adjacency"],self.data[idx]["edge_order"],self.data[idx]["tree_sizes"],self.data[idx]["feature_label"],self.data[idx]["tokenseq_len"],self.data[idx]["sen_labels"],self.data[idx]["len_sen"],self.data[idx]["this_name"]

    def __len__(self):  # What is the length of the dataset
        return len(self.data)
    # def embedding_word(self,word):
    #     index = self.word2idx[word]
    #     embedding = self.embedding_weights[index]
    #     return embedding
    def embedding_Tword(self,word):
        try:
            index = self.Tword2idx[word]
        except:
            index = self.Tword2idx['<UNK>']
        embedding = self.Tembedding_weights[index]
        return embedding
    def idx_word(self,word):
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

        all_tokenList = list()
        all_seqlen = list()
        for limit in range(limit_num):
            for token_seqs in de.loc[num, ["codeSquence"]].values:
                all_seqlen.append(len(token_seqs))
                for sens in token_seqs:
                    token_seq = list()
                    for token in sens:
                            token_seq.append(token)
                    all_tokenList.append(token_seq)

        return all_tokenList,all_seqlen
    def getNewTree(self,limit_num, num, de):

        all_treeList1 = list()
        for limit in range(limit_num):
            all_treeList = list()
            for token_seqs in de.loc[num, ["node_value"]].values:
                for sens in token_seqs:
                    token_seq = list()
                    for token in sens:
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
    def pad_sequence2(self,sequences, padding_value):
        max_len = max([s.size(0) for s in sequences])
        out_tensor = torch.zeros(len(sequences),max_len)
        for i, tensor in enumerate(sequences):
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :] = padding_value

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :length] = tensor


        return out_tensor
    def getOtherThree(self,limit_num, num, de):
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
        return adjacency_lists, node_orders, edge_orders, seq_labels,sen_labels
def mao_pao(l):
    for i in range(len(l)):
        for j in range(len(l)-i-1):
            if l[j][0]<l[j+1][0]:
                l[j],l[j+1] = l[j+1],l[j]
    return l
def test(model, iterator):
    result_list = list()

    for i, (trg, features, node_order, adjacency_list, edge_order,treesizes,feature_label,tokenseq_len,sen_label,sen_len,this_name) in enumerate(iterator):
        data_dict1 = {
            "rank_list": None,
            "this_name": None,
            "top_1":None,
            "top_3":None,
            "top_5": None,
            "MAR":None,
            "MFR":None,
            "exam":None

        }
        trg = trg.reshape(trg.shape[1],trg.shape[2])
        trg = trg.permute(1, 0)
        features = features.reshape(features.shape[1],features.shape[2])
        node_order = node_order.reshape(node_order.shape[1])
        adjacency_list = adjacency_list.reshape(adjacency_list.shape[1],adjacency_list.shape[2])
        edge_order = edge_order.reshape(edge_order.shape[1])
        output,output1,aa = model.forward(features, node_order, adjacency_list, edge_order, treesizes, trg, tokenseq_len,sen_len)
        cccc = aa.tolist()
        aa_list = list()
        sl_list = list()

        for a in cccc:
            for b in a:
                for c in b:
                    aa_list.append(c)
        sss = sen_label.tolist()
        for a in sss:
            for b in a:
                for c in b:
                    sl_list.append(c)
        al_lists = list()
        if len(sl_list)!=len(aa_list):
            print(aa.shape)
            print(sen_label.shape)
            print(this_name)
            continue
        else:
            for a in range(len(aa_list)):
                al_list = list()
                al_list.append(aa_list[a])
                al_list.append(sl_list[a])
                al_lists.append(al_list)
        ll = mao_pao(al_lists)
        top1 = None
        top3 = None
        top5 = None

        if int(ll[0][1]) == 1:
            top1 = 1
        if len(ll)>3:
            if int(ll[0][1]) == 1 or int(ll[1][1]) == 1 or int(ll[2][1]) == 1:
                top3 = 1
        else:
            top3 = 1
        if len(ll)>5:
            if int(ll[0][1]) == 1 or int(ll[1][1]) == 1 or int(ll[2][1]) == 1 or int(ll[3][1]) == 1 or int(ll[4][1]) == 1:
                top5 = 1
        else:
            top5 = 1
        count = 0
        f_flag = 0
        mar = 0
        data_dict1["exam"] = len(ll)
        for a in range(len(ll)):
            if int(ll[a][1]) == 1 and f_flag == 0 :
                data_dict1["MFR"] = a+1

                mar += a + 1
                f_flag = 1
                count += 1
            elif int(ll[a][1]) == 1 and f_flag != 0:
                mar += a+1
                count += 1

        try:
            data_dict1["MAR"] = '%.2f' %(mar/count)
        except Exception:

            print(this_name)

        data_dict1["rank_list"] = ll
        data_dict1["top_1"] = top1
        data_dict1["top_3"] = top3
        data_dict1["top_5"] = top5
        data_dict1["this_name"] = this_name

        result_list.append(data_dict1)

    return result_list
if __name__ == '__main__':
    with open(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\idx2word.pkl', 'rb') as f:
        idx2word = pickle.load(f)
    with open(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\word2idx.pkl', 'rb') as f1:
        word2idx = pickle.load(f1)
    MAX_VOCAB_SIZE = 10000
    EMBEDDING_SIZE = 512
    model1 = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    model1.load_state_dict(torch.load(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\embedding-512.th'))
    embedding_weights = model1.input_embedding()
    t_top1 = 0
    t_top3 = 0
    t_top5 = 0
    T_MAR = 0
    T_MFR = 0
    exam_d = data_d = {
        "10": 0,
        "20": 0,
        "30": 0,
        "40": 0,
        "50": 0,
        "60": 0,
        "70": 0,
        "80": 0,
        "90": 0,
        "100": 0,

    }
    rrr = 0
    for ccccc in range(10):
        limit_num = 1
        NUM = ccccc
        # name = "Lang"
        b = np.load(r"E:\pycharmworkSpace\StatementContrastive\a"+str(NUM)+".npy")
        # b = np.load(r"E:\pycharmworkSpace\StatementContrastive\cross_project\\"+ name + ".npy")
        b = b.tolist()
        dataset2 = pro_DataSet(r"E:\pycharmworkSpace\StatementContrastive\data\data_new1.json",limit_num,b)  # create the dataset
        dataloader2 = torch.utils.data.DataLoader(dataset=dataset2, shuffle=True, batch_size=1,drop_last=True)

        device = torch.device('cpu')

        # model = torch.load(r"E:\pycharmworkSpace\StatementContrastive\model\2\tut"+str(NUM)+"-model.pt",map_location='cpu')
        model = torch.load(r"E:\pycharmworkSpace\StatementContrastive\model\identifier\e-30-a" + str(NUM) + ".pt",
                           map_location='cpu')
        # model = torch.load(r"E:\pycharmworkSpace\StatementContrastive\model\cross_project\cross-" + name + "-model.pt",map_location='cpu')
        output = test(model, dataloader2)
        d_list = list()


        for iii in output:
            data_d = {
                "this_name": None,
                "top_1": None,
                "top_3": None,
                "top_5": None,
                "exam":None
            }
            # print("projectname: ",iii["this_name"]," |ranklist: ", iii["rank_list"]," |top-1: ",iii["top_1"]," |top-3: ",iii["top_3"]," |top-5: ",iii["top_5"])
            # print("projectname: ", iii["this_name"], " |top-1: ", iii["top_1"]," |top-3: ", iii["top_3"], " |top-5: ", iii["top_5"])
            # print(iii["rank_list"])
            data_d["this_name"] = iii["this_name"]
            flag = 0
            if d_list:
                for a in d_list:
                    if a["this_name"] == iii["this_name"]:
                        flag = 1
                        if a["top_1"] == None:
                            a["top_1"] = iii["top_1"]
                        if a["top_3"] == None:
                            a["top_3"] = iii["top_3"]
                        if a["top_5"] == None:
                            a["top_5"] = iii["top_5"]
                        if a["MAR"]>iii["MAR"]:
                            a["MAR"] = iii["MAR"]
                        if a["MFR"]>iii["MFR"]:
                            a["MFR"] = iii["MFR"]
                            a["exam"] = iii["exam"]
                        break
            if flag == 0:
                data_d["this_name"] = iii["this_name"]
                data_d["top_1"] = iii["top_1"]
                data_d["top_3"] = iii["top_3"]
                data_d["top_5"] = iii["top_5"]
                data_d["MAR"] = iii["MAR"]
                data_d["MFR"] = iii["MFR"]
                data_d["exam"] = iii["exam"]
                d_list.append(data_d)
        top1 = 0
        top3 = 0
        top5 = 0
        COUNT = 0
        MAR1 = float(0)
        MFR1 = float(0)
        exam = 0

        for iii in d_list:
            # print("projectname: ", iii["this_name"], " |top-1: ", iii["top_1"], " |top-3: ", iii["top_3"], " |top-5: ",
            #       iii["top_5"])
            COUNT+=1
            if iii["top_1"] == 1:
                top1+=1
            if iii["top_3"] == 1:
                top3+=1
            if iii["top_5"] == 1:
                top5+=1
            MAR1 += float(iii["MAR"])
            MFR1 += float(iii["MFR"])
            exam = float(iii["MFR"]/iii["exam"])
            rrr += iii["exam"]
            if 0<exam and exam<=0.10:
                exam_d["10"] += 1
            if 0<exam and exam<=0.20:
                exam_d["20"] += 1
            if 0<exam and exam<=0.30:
                exam_d["30"] += 1
            if 0<exam and exam<=0.40:
                exam_d["40"] += 1
            if 0<exam and exam<=0.50:
                exam_d["50"] += 1
            if 0<exam and exam<=0.60:
                exam_d["60"] += 1
            if 0<exam and exam<=0.70:
                exam_d["70"] += 1
            if 0<exam and exam<=0.80:
                exam_d["80"] += 1
            if 0<exam and exam<=0.90:
                exam_d["90"] += 1
            if 0<exam and exam<=1:
                exam_d["100"]+=1
        t_top1 += top1
        t_top3 += top3
        t_top5 += top5
        T_MAR += MAR1/COUNT
        T_MFR += MFR1/COUNT


    print("total: ")
    print("top1: ",t_top1,"  |top3: ",t_top3,"  |top5: ",t_top5,"  |MAR: ",'%.2f' %(T_MAR),"  |MFR: ",'%.2f' %(T_MFR))
    for k,v in exam_d.items():
        print(k,":  ",float(v/381))
    print("总共")
    print(rrr)



# 0 :top1:  5   |top3:  19   |top5:  21   |MAR:  9.22   |MFR:  7.10
# 1 :top1:  12   |top3:  20   |top5:  25   |MAR:  10.88   |MFR:  8.64
# 2 :top1:  12   |top3:  19   |top5:  24   |MAR:  11.72   |MFR:  9.33
# 3 :top1:  18   |top3:  22   |top5:  25   |MAR:  8.73   |MFR:  6.74
# 4 :top1:  14   |top3:  21   |top5:  22   |MAR:  17.09   |MFR:  7.78
# 5 :top1:  12   |top3:  13   |top5:  17   |MAR:  13.41   |MFR:  9.89
# 6 :top1:  11   |top3:  19   |top5:  22   |MAR:  10.45   |MFR:  7.97
# 7 :top1:  12   |top3:  17   |top5:  21   |MAR:  11.77   |MFR:  8.83
# 8 :top1:  9   |top3:  19   |top5:  24   |MAR:  10.81   |MFR:  7.15
# 9 :top1:  17   |top3:  23   |top5:  27   |MAR:  7.61   |MFR:  6.45

#total:
#:top1:  122   |top3:  192   |top5: 228  |MAR:  11.17   |MFR:  7.99  |p%:30.89

# 0 :top1:  2   |top3:  18   |top5:  22   |MAR:  9.62   |MFR:  7.10
# 1 :top1:  9   |top3:  23   |top5:  27   |MAR:  8.26   |MFR:  6.92
# 2 :top1:  11   |top3:  17   |top5:  19   |MAR:  13.71   |MFR:  9.72
# 3 :top1:  15   |top3:  20   |top5:  28   |MAR:  8.27   |MFR:  7.21
# 4 :top1:  12   |top3:  17   |top5:  20   |MAR:  15.44   |MFR:  11.32
# 5 :top1:  9   |top3:  14   |top5:  17   |MAR:  12.76   |MFR:  10.14
# 6 :top1:  11   |top3:  20   |top5:  24   |MAR:  11.40   |MFR:  9.58
# 7 :top1:  7   |top3:  15   |top5:  20   |MAR:  10.91   |MFR:  9.08
# 8 :top1:  11   |top3:  16   |top5:  21   |MAR:  12.60   |MFR:  7.72
# 9 :top1:  10   |top3:  18   |top5:  20   |MAR:  8.36   |MFR:  7.50
#total:
#:top1:  97   |top3:  178   |top5: 218  |MAR:  11.13   |MFR:  8.62  |p%:24.56
