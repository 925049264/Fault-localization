import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
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
        outputs = self.relu(outputs)
        outpute, aa = self.decoder.forward(outputs,token_lenth)
        # print(type())
        # print(type(hidden))
        # print(type(output))
        y = self.liner2(outpute)
        outputs11 = torch.sigmoid(y)
        return outpute,outputs11, aa