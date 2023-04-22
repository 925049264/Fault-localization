import copy
import os
import random
import time

import numpy as np
from torch import optim

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn as rnn_utils
import pickle
import pandas as pd
import torch.nn.functional as F
class pro_DataSet(torch.utils.data.Dataset):
    def __init__(self, filename, pro_num):

        de = pd.read_json(filename, lines=True)
        # 给每个token嵌入词向量
        data_list = list()

        # for i in range(4):
        for i in range(len(de)):

            if i % 2 == 1:
                p_num = None
                p_name = None
                covers = None
                sen_labels = None
                for a in de.loc[i, ["pro_num"]].values:
                    p_num = a
                for a in de.loc[i, ["pro_name"]].values:
                    p_name = a
                for a in de.loc[i, ["codecover"]].values:
                    covers = a
                for a in de.loc[i, ["sen_label"]].values:
                    sen_labels = a
                this_name = str(p_name) + str(p_num)
                if this_name in pro_num:
                    print(i)
                    continue

                for a in range(len(covers)):
                    data_dict = {
                        "token_seq": None,
                        # "tokenseq_len": None,
                        "cover": None,
                        "mutant": None,
                        "sen_labels": None
                    }
                    cc = covers[a].split(",")

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
                    # data_dict["tokenseq_len"] =tokenseq_len
                    data_dict["cover"] = cover
                    data_dict["sen_labels"] = sen_label
                    data_list.append(data_dict)
        self.data = data_list

    def __getitem__(self, idx):  # if the index is idx, what will be the data?
        return self.data[idx]["cover"], self.data[idx]["sen_labels"]

    def __len__(self):  # What is the length of the dataset
        return len(self.data)
class CNN1(nn.Module):
    def __init__(self, in_channel=1, feature=15, num_class=512, num_class1=1):
        super(CNN1, self).__init__()
        self.Conv1 = nn.Conv1d(in_channel, feature, kernel_size=3, stride=1, padding=1)
        self.Relu1 = nn.ReLU()
        self.Pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Fc = nn.Linear(3840, num_class)
        self.Relu2 = nn.ReLU()
        self.Fc1 = nn.Linear(num_class, num_class1)



    def forward(self, x):
        x = self.Conv1(x)
        x = self.Relu1(x)
        x = self.Pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Fc(x)
        x = self.Relu2(x)
        x = self.Fc1(x)
        x = torch.sigmoid(x)
        return x

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


    def train(model, iterator, optimizer, criterion, clip, batch_s):

        # model.train()
        epoch_loss = 0
        for i, (cover, sen_label) in enumerate(iterator):
            cover =  torch.tensor([item.cpu().detach().numpy() for item in cover],dtype=torch.float)
            cover = cover.unsqueeze(1)
            cover = torch.reshape(cover,(cover.shape[2],cover.shape[1],cover.shape[0])).to(device)

            optimizer.zero_grad()

            output = model.forward(cover)

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
    fmodel = CNN1(in_channel=1, feature=15, num_class=32).to(device)

    # b = np.load('a3.npy')
    # b = b.tolist()
    # dataset1 = pro_DataSet(r"data/data_new1.json",pro_num=b)  # create the dataset
    dataset1 = pro_DataSet(r"data/data_new3.json", pro_num=[])  # create the dataset
    dataloader = torch.utils.data.DataLoader(dataset=dataset1, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    fmodel.apply(init_weights)
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