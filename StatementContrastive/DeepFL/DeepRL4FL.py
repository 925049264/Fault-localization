import copy
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import time

import numpy as np
from torch import optim
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn as rnn_utils
import pickle
import pandas as pd
import torch.nn.functional as F


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


class pro_DataSet(torch.utils.data.Dataset):
    def __init__(self, filename, pro_num):
        # with open(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\idx2word.pkl', 'rb') as f:
        #     self.idx2word = pickle.load(f)
        # with open(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\word2idx.pkl', 'rb') as f1:
        #     self.word2idx = pickle.load(f1)
        # MAX_VOCAB_SIZE = 10000
        # EMBEDDING_SIZE = 512
        # model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
        # model.load_state_dict(torch.load(r'E:\pycharmworkSpace\StatementContrastive\identifier\token_vec\embedding-512.th'))
        # self.embedding_weights = model.input_embedding()
        de = pd.read_json(filename, lines=True)
        # 给每个token嵌入词向量
        data_list = list()

        # for i in range(4):
        for i in range(len(de)):

            if i % 2 == 1:
                p_num = None
                p_name = None
                codeSquence = None
                covers = None
                mutants = None
                feature_labels = None
                sen_labels = None
                for a in de.loc[i, ["pro_num"]].values:
                    p_num = a
                for a in de.loc[i, ["pro_name"]].values:
                    p_name = a
                for a in de.loc[i, ["codeSquence"]].values:
                    codeSquence = a
                for a in de.loc[i, ["Mutant_cover1"]].values:
                    mutants = a
                for a in de.loc[i, ["codecover"]].values:
                    covers = a
                # for a in de.loc[i, ["seq_label"]].values:
                #     feature_labels = a
                for a in de.loc[i, ["sen_label"]].values:
                    sen_labels = a
                this_name = str(p_name) + str(p_num)
                if this_name in pro_num:
                    print(i)
                    continue

                for a in range(len(codeSquence)):
                    data_dict = {
                        "token_seq": None,
                        # "tokenseq_len": None,
                        "cover": None,
                        "mutant": None,
                        "sen_labels": None
                    }
                    token_seq = list()
                    for b in range(8):
                        try:
                            token_seq.append(codeSquence[a][b])
                        except:
                            token_seq.append("<pad>")

                    tokenseq_len = len(token_seq)
                    #                     print(covers[a])
                    cc = covers[a].split(",")

                    mutant = list()

                    if len(mutants) < 32:
                        for b in range(len(mutants)):
                            lst = list()

                            for c in range(512):
                                try:
                                    lst.append(float(mutants[b][a][c]))
                                except:
                                    lst.append(0)
                            mutant.append(lst)
                        for b in range(32 - len(mutants)):
                            lst = list()
                            for c in range(512):
                                try:
                                    lst.append(float(mutants[b][a][c]))
                                except:
                                    lst.append(0)
                            mutant.append(lst)
                    else:
                        for b in range(32):
                            lst = list()

                            for c in range(512):
                                try:
                                    lst.append(float(mutants[b][a][c]))
                                except:
                                    lst.append(0)
                            mutant.append(lst)

                    cover = list()
                    for b in range(512):
                        try:
                            if cc[b] == "1":
                                cover.append(1)
                            elif cc[b] == "-1":
                                cover.append(-1)
                            else:
                                cover.append(0)
                        except:
                            cover.append(0)

                    sen_label = np.array(sen_labels[a])
                    data_dict["token_seq"] = token_seq
                    # data_dict["tokenseq_len"] =tokenseq_len
                    data_dict["cover"] = cover
                    data_dict["mutant"] = np.array(mutant)
                    data_dict["sen_labels"] = sen_label
                    data_list.append(data_dict)
        self.data = data_list

    def __getitem__(self, idx):  # if the index is idx, what will be the data?
        return self.data[idx]["token_seq"], self.data[idx]["cover"], self.data[idx]["mutant"], self.data[idx][
            "sen_labels"]

    def __len__(self):  # What is the length of the dataset
        return len(self.data)


class CNN2(nn.Module):
    def __init__(self, in_channel=1, feature=15, num_class=32):
        super(CNN2, self).__init__()
        self.Conv1 = nn.Conv2d(in_channel, feature, kernel_size=3, stride=1, padding=1)
        self.Relu1 = nn.ReLU()
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Fc = nn.Linear(61440, num_class)

    def forward(self, x):
        x = self.Conv1(x)  # [B,1,32,512] --> [B,15,32,512]
        x = self.Relu1(x)
        x = self.Pool1(x)  # [B,15,32,512] --> [B,15,16,256]
        x = x.reshape(x.shape[0], -1)
        x = self.Fc(x)
        return x


class CNN1(nn.Module):
    def __init__(self, in_channel=1, feature=15, num_class=32):
        super(CNN1, self).__init__()
        self.Conv1 = nn.Conv1d(in_channel, feature, kernel_size=3, stride=1, padding=1)
        self.Relu1 = nn.ReLU()
        self.Pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Fc = nn.Linear(3840, num_class)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Relu1(x)
        x = self.Pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, num_input, num_hidden, num_class=32):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_class)
        self.Relu1 = nn.ReLU()
        self.Relu2 = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.Relu1(x)
        x = self.linear2(x)
        x = self.Relu2(x)
        return x


class FullModel(nn.Module):
    def __init__(self, CNN1, CNN2, MLP, in_channel=1, feature=15, num_class=1):
        super().__init__()
        self.CNN1 = CNN1
        self.CNN2 = CNN2
        self.MLP = MLP

        self.Conv1 = nn.Conv3d(in_channel, feature, kernel_size=3, stride=1, padding=1)
        self.Relu1 = nn.ReLU()
        self.Pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Fc = nn.Linear(61440, 1)

    def forward(self, cover, mutant, word):
        cov = self.CNN1.forward(cover)
        mut = self.CNN2.forward(mutant)
        wo = self.MLP.forward(word)
        cov = cov.unsqueeze(1).unsqueeze(1)
        mut = mut.unsqueeze(1).unsqueeze(1)
        wo = wo.unsqueeze(1).unsqueeze(1)
        cov = torch.reshape(cov, (cov.shape[0], cov.shape[1], cov.shape[3], cov.shape[2]))
        mut = torch.reshape(mut, (mut.shape[0], mut.shape[3], mut.shape[1], mut.shape[2]))

        c = torch.mul(cov, mut)

        b = torch.mul(c, wo)  # [15, 1, 3, 3, 3]
        b = b.unsqueeze(1)

        x = self.Conv1(b)

        x = self.Relu1(x)
        x = self.Pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Fc(x)
        x = torch.sigmoid(x)
        return x


# self.data[idx]["token_seq"],self.data[idx]["cover"],self.data[idx]["mutant"],self.data[idx]["sen_labels"]

if __name__ == '__main__':
    device = torch.device('cuda:0')


    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    with open(r'data/token_vec/idx2word.pkl', 'rb') as f2:
        Tidx2word = pickle.load(f2)
    with open(r'data/token_vec/word2idx.pkl', 'rb') as f3:
        Tword2idx = pickle.load(f3)
    MAX_VOCAB_SIZE = 10000
    EMBEDDING_SIZE = 512
    model1 = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    model1.load_state_dict(torch.load(r'data/token_vec/embedding-512.th'))
    Tembedding_weights = model1.input_embedding()


    def embedding_Tword(word):
        if word == "<pad>":
            embedding = torch.zeros(512)
        else:
            try:
                index = Tword2idx[word]
            except:
                index = Tword2idx['<UNK>']
            embedding = torch.tensor(Tembedding_weights[index])
        return embedding


    def train(model, iterator, optimizer, criterion, clip, batch_s):

        # model.train()
        epoch_loss = 0
        for i, (token_seq, cover, mutant, sen_label) in enumerate(iterator):
            cover = torch.tensor([item.cpu().detach().numpy() for item in cover], dtype=torch.float)
            cover = cover.unsqueeze(1)
            cover = torch.reshape(cover, (cover.shape[2], cover.shape[1], cover.shape[0])).to(device)
            mutant = mutant.unsqueeze(1)
            mutant = mutant.float().to(device)

            #             mutant = torch.Tensor(mutant)
            #             sen_label = torch.Tensor(sen_label)
            #             sen_label = torch.from_numpy(sen_label)
            tss = list()

            for b in range(batch_s):
                ts = None
                for seq in range(len(token_seq)):
                    if seq == 0:
                        ts = embedding_Tword(token_seq[seq][b])
                    else:
                        ts = torch.cat((ts, embedding_Tword(token_seq[seq][b])), dim=-1)
                tss.append(ts)
            token_seq = torch.tensor([item.cpu().detach().numpy() for item in tss], dtype=torch.float).to(device)
            optimizer.zero_grad()

            output = model.forward(cover, mutant, token_seq)

            sen_label = sen_label.unsqueeze(1).float().to(device)

            loss = criterion.forward(output, sen_label)

            loss.backward()

            # gradient clipping 防止梯度爆炸问题
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(iterator)


    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    BATCH_SIZE = 2
    c1 = CNN1(in_channel=1, feature=15, num_class=32).to(device)
    c2 = CNN2(in_channel=1, feature=15, num_class=32).to(device)
    mlp = MLP(4096, 512, 32).to(device)
    fmodel = FullModel(c1, c2, mlp, in_channel=1, feature=15, num_class=2).to(device)
    # b = np.load('a3.npy')
    # b = b.tolist()
    # dataset1 = pro_DataSet(r"data/data_new1.json",pro_num=b)  # create the dataset
    dataset1 = pro_DataSet(r"data/data_new3.json", pro_num=[])  # create the dataset
    dataloader = torch.utils.data.DataLoader(dataset=dataset1, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    # fmodel.apply(init_weights)
    optimizer = optim.Adam(fmodel.parameters())
    N_EPOCHS = 10
    CLIP = 1
    criterion = nn.BCELoss()
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(fmodel, dataloader, optimizer, criterion, CLIP, BATCH_SIZE)
        # valid_loss = evaluate(model, dataloader1, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        torch.save(fmodel, 'tut3-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.6f}')


