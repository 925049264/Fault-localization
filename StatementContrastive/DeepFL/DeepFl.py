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


class formula():
    def __init__(self, formula_name, tf, tp, Ftf, Ftp, l):
        self.formula_name = formula_name
        self.tp = float(tp)
        self.tf = float(tf)
        self.Ftp = float(Ftp)
        self.Ftf = float(Ftf)
        self.mu = l

    def getScore(self):
        if self.formula_name == "Tarantula":
            self.Tarantula(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Ample":
            self.Ample(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "SorensenDice":
            self.SorensenDice(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Kulczynski2":
            self.Kulczynski2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "M1":
            self.M1(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Goodman":
            self.Goodman(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Overlap":
            self.Overlap(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "zoltar":
            self.zoltar(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "ER5c":
            self.ER5c(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "GP13":
            self.GP13(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "DStar2":
            self.DStar2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "DStar":
            self.DStar(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "ER1a":
            self.ER1a(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "ER1b":
            self.ER1b(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Wong3":
            self.Wong3(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "GP19":
            self.GP19(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "GP02":
            self.GP02(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Wong1":
            self.Wong1(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Anderberg":
            self.Anderberg(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Hamming":
            self.Hamming(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "M2":
            self.M2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "SimpleMatching":
            self.SimpleMatching(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Dice":
            self.Dice(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "RussellRao":
            self.RussellRao(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Ochiai":
            self.Ochiai(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Jaccard":
            self.Jaccard(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Hamann":
            self.Hamann(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Kulczynski1":
            self.Kulczynski1(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Sokal":
            self.Sokal(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "RogersTanimoto":
            self.RogersTanimoto(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Euclid":
            self.Euclid(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Ochiai2":
            self.Ochiai2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Wong2":
            self.Wong2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "GP03":
            self.GP03(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "SBI":
            self.SBI(self.tf, self.tp, self.Ftf, self.Ftp)

    def getAllScore(self):
        slist = list()
        slist.append(self.Tarantula(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Ample(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.SorensenDice(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Kulczynski2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.M1(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Goodman(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Overlap(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.zoltar(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.ER5c(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.GP13(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.DStar2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.ER1a(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.ER1b(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Wong3(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.GP19(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.GP02(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Wong1(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Anderberg(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Hamming(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.M2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.SimpleMatching(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Dice(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.RussellRao(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Ochiai(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Jaccard(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Hamann(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Kulczynski1(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Sokal(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.RogersTanimoto(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Euclid(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Ochiai2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Wong2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.GP03(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.SBI(self.tf, self.tp, self.Ftf, self.Ftp))
        return slist

    def getAllScore1(self, tf, tp):
        tf = float(tf)
        tp = float(tp)
        slist = list()
        slist.append(self.Tarantula(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Ample(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.SorensenDice(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Kulczynski2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.M1(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Goodman(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Overlap(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.zoltar(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.ER5c(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.GP13(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.DStar2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.ER1a(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.ER1b(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Wong3(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.GP19(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.GP02(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Wong1(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Anderberg(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Hamming(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.M2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.SimpleMatching(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Dice(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.RussellRao(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Ochiai(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Jaccard(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Hamann(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Kulczynski1(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Sokal(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.RogersTanimoto(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Euclid(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Ochiai2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Wong2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.GP03(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.SBI(tf, tp, self.Ftf, self.Ftp))
        return slist

    def getMutantScore(self, l):
        list1 = list()
        for i in range(len(l)):
            list1.append(self.getAllScore1(l[i][0], l[i][1]))
        list2 = list()
        for i in range(34):
            num = -10000000
            for j in range(len(l)):
                if list1[j][i] >= num:
                    num = list1[j][i]
            list2.append(num)
        num1 = float(0)
        for i in range(len(l)):
            num1 += self.MUSE(l[i][0], l[i][1], self.Ftf, self.Ftp)
        a = float(num1 / float(len(l)))
        list2.append(a)
        return list2

    def MUSE(self, tf, tp, Ftf, Ftp):
        tf = float(tf)
        tp = float(tp)
        Ftf = float(Ftf)
        Ftp = float(Ftp)
        if (tf + Ftf) == 0 and (tp + Ftp) != 0:
            a = - tp / (tp + Ftp)
        elif (tf + Ftf) != 0 and (tp + Ftp) == 0:
            a = tf / (tf + Ftf)
        elif (tf + Ftf) == 0 and (tp + Ftp) == 0:
            a = 0
        else:
            a = tf / (tf + Ftf) - tp / (tp + Ftp)
        return a

    def Tarantula(self, tf, tp, Ftf, Ftp):
        if Ftf + tf == 0:
            a = 0
        else:
            a = tf / (Ftf + tf)
        if Ftp + tp == 0:
            b = 0
        else:
            b = tp / (Ftp + tp)
        if a + b == 0:
            s = 0
        else:
            s = a / (a + b)
        return s

    def Ample(self, tf, tp, Ftf, Ftp):
        if Ftf + tf == 0:
            a = 0
        else:
            a = tf / (Ftf + tf)
        if Ftp + tp == 0:
            b = 0
        else:
            b = tp / (Ftp + tp)
        s = a - b
        return s

    def SorensenDice(self, tf, tp, Ftf, Ftp):
        if tf + tp + Ftf == 0:
            s = 0
        else:
            s = (2 * tf) / (2 * tf + tp + Ftf)
        return s

    def Kulczynski2(self, tf, tp, Ftf, Ftp):
        if Ftf + tf == 0:
            a = 0
        else:
            a = tf / (Ftf + tf)
        if tf + tp == 0:
            b = 0
        else:
            b = tf / (tf + tp)
        s = 0.5 * (a + b)
        return s

    def M1(self, tf, tp, Ftf, Ftp):
        if Ftf + tp == 0:
            s = 0
        else:
            s = (tf + Ftp) / (Ftf + tp)
        return s

    def Goodman(self, tf, tp, Ftf, Ftp):
        if (tf + Ftf + tp) == 0:
            s = 0
        else:
            s = (2 * tf - Ftf - tp) / (2 * tf + Ftf + tp)
        return s

    def Overlap(self, tf, tp, Ftf, Ftp):
        a = min(tf, tp, Ftf)
        if a == 0:
            s = 0
        else:
            s = tf / a
        return s

    def zoltar(self, tf, tp, Ftf, Ftp):
        if tf == 0:
            return 0
        else:
            a = tf + Ftf + tp + ((10000 * Ftf * tp) / tf)
            if a == 0:
                return 0
            else:
                return tf / a

    def ER5c(self, tf, tp, Ftf, Ftp):
        if tf < tf + Ftf:
            return 0
        else:
            return 1

    def GP13(self, tf, tp, Ftf, Ftp):
        if tp + tf == 0:
            return 0
        else:
            s = tf * (1 + (1 / (2 * tp + tf)))
            return s

    def DStar2(self, tf, tp, Ftf, Ftp):
        if tp + Ftf == 0:
            return 0
        else:
            return (tf * tf) / (tp + Ftf)

    def DStar(self, tf, tp, Ftf, Ftp):
        if tp + Ftf == 0:
            return 0
        else:
            return tf / tp + Ftf

    def ER1a(self, tf, tp, Ftf, Ftp):
        if tf < Ftf + tf:
            return -1
        else:
            return Ftp

    def ER1b(self, tf, tp, Ftf, Ftp):
        return tf - tp / (tp + Ftp + 1)

    def Wong3(self, tf, tp, Ftf, Ftp):
        if tp <= 2:
            return tf - tp
        elif tp <= 10 and tp > 2:
            return tf - 2 - 0.1 * (tp - 2)
        else:
            return tf - 2.8 - 0.01 * (tp - 10)

    def GP19(self, tf, tp, Ftf, Ftp):
        a = tp - tf + tf + Ftf - tp - Ftp
        return tf * pow(a, 0.5)

    def GP02(self, tf, tp, Ftf, Ftp):
        return 2 * (tf + tp + Ftp) + tp

    def Wong1(self, tf, tp, Ftf, Ftp):
        return tf

    def Anderberg(self, tf, tp, Ftf, Ftp):
        a = tf + 2 * Ftf + 2 * tp
        if a == 0:
            return 0
        else:
            return tf / a

    def Hamming(self, tf, tp, Ftf, Ftp):
        return tf + Ftp

    def M2(self, tf, tp, Ftf, Ftp):
        a = tf + Ftp + 2 * Ftf + 2 * tp
        if a == 0:
            return 0
        else:
            return tf / a

    def SimpleMatching(self, tf, tp, Ftf, Ftp):
        a = tf + tp + Ftf + Ftp
        if a == 0:
            return 0
        else:
            return (tf + Ftp) / a

    def Dice(self, tf, tp, Ftf, Ftp):
        a = tf + Ftf + tp
        if a == 0:
            return 0
        else:
            return 2 * tf / a

    def RussellRao(self, tf, tp, Ftf, Ftp):
        a = tf + tp + Ftf + Ftp
        if a == 0:
            return 0
        else:
            return tf / a

    def Ochiai(self, tf, tp, Ftf, Ftp):
        a = (tf + tp) * (tf + Ftf)
        if a == 0:
            return 0
        else:
            return tf / a

    def Jaccard(self, tf, tp, Ftf, Ftp):
        a = tf + Ftf + tp
        if a == 0:
            return 0
        else:
            return tf / a

    def Hamann(self, tf, tp, Ftf, Ftp):
        a = tf + tp + Ftf + Ftp
        if a == 0:
            return 0
        else:
            return (tf + Ftp - tp - Ftf) / a

    def Kulczynski1(self, tf, tp, Ftf, Ftp):
        a = Ftp + Ftf + tp
        if a == 0:
            return 0
        else:
            return tf / a

    def Sokal(self, tf, tp, Ftf, Ftp):
        a = 2 * tf + 2 * Ftp + Ftp + tp
        if a == 0:
            return 0
        else:
            return (2 * tf + 2 * Ftp) / a

    def RogersTanimoto(self, tf, tp, Ftf, Ftp):
        a = tf + Ftp + 2 * Ftp + 2 * tp
        if a == 0:
            return 0
        else:
            return (tf + Ftp) / a

    def Euclid(self, tf, tp, Ftf, Ftp):
        n = tf + Ftp
        return pow(n, 0.5)

    def Ochiai2(self, tf, tp, Ftf, Ftp):
        a = (tf + tp) * (Ftf + Ftp) * (tf + Ftp) * (Ftf + tp)
        if a == 0:
            return 0
        else:
            return (tf + tp) / a

    def Wong2(self, tf, tp, Ftf, Ftp):
        return tf - tp

    def GP03(self, tf, tp, Ftf, Ftp):
        a = tf * tf - pow(tp, 0.5)
        return pow(a, 0.5)

    def SBI(self, tf, tp, Ftf, Ftp):
        a = tf + tp
        if a == 0:
            return 0
        else:
            return tf / a


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
                code_info = None
                mutant_info1 = None
                mutant_info2 = None
                mutant_info3 = None
                mutant_info4 = None

                sen_labels = None
                for a in de.loc[i, ["pro_num"]].values:
                    p_num = a
                for a in de.loc[i, ["pro_name"]].values:
                    p_name = a

                for a in de.loc[i, ["Mutant_info1"]].values:
                    mutant_info1 = a
                for a in de.loc[i, ["Mutant_info2"]].values:
                    mutant_info2 = a
                for a in de.loc[i, ["Mutant_info3"]].values:
                    mutant_info3 = a
                for a in de.loc[i, ["Mutant_info4"]].values:
                    mutant_info4 = a
                for a in de.loc[i, ["codeinfo"]].values:
                    code_info = a
                for a in de.loc[i, ["sen_label"]].values:
                    sen_labels = a
                this_name = str(p_name) + str(p_num)
                if this_name in pro_num:
                    print(i)
                    continue

                for a in range(len(code_info)):
                    data_dict = {
                        "Spectra": None,
                        "mutant1": None,
                        "mutant2": None,
                        "mutant3": None,
                        "mutant4": None,
                        "sen_labels": None
                    }
                    codeinfo = code_info[a].split(",")  # [[tf,tp,-tf,-tp]]
                    tf = int(codeinfo[0])
                    tp = int(codeinfo[0])
                    Ftf = int(codeinfo[0])
                    Ftp = int(codeinfo[0])

                    cinfo_f = formula("", tf, tp, Ftf, Ftp, [])
                    cinfo = cinfo_f.getAllScore()

                    mutantinfo1 = mutant_info1[a]
                    tf1 = 1
                    tp1 = 1
                    minfo1 = formula("", tf1, tp1, Ftf, Ftp, mutantinfo1).getMutantScore(mutantinfo1)

                    mutantinfo2 = mutant_info2[a]
                    tf2 = 1
                    tp2 = 1
                    minfo2 = formula("", tf2, tp2, Ftf, Ftp, mutantinfo2).getMutantScore(mutantinfo2)

                    mutantinfo3 = mutant_info3[a]
                    tf3 = 1
                    tp3 = 1
                    minfo3 = formula("", tf3, tp3, Ftf, Ftp, mutantinfo3).getMutantScore(mutantinfo3)

                    mutantinfo4 = mutant_info4[a]
                    tf4 = 1
                    tp4 = 1
                    minfo4 = formula("", tf4, tp4, Ftf, Ftp, mutantinfo4).getMutantScore(mutantinfo4)
                    sen_label = np.array(sen_labels[a])

                    data_dict["Spectra"] = np.array(cinfo)
                    data_dict["mutant1"] = np.array(minfo1)
                    data_dict["mutant2"] = np.array(minfo2)
                    data_dict["mutant3"] = np.array(minfo3)
                    data_dict["mutant4"] = np.array(minfo4)
                    data_dict["sen_labels"] = sen_label
                    data_list.append(data_dict)
        self.data = data_list

    def __getitem__(self, idx):  # if the index is idx, what will be the data?
        return self.data[idx]["Spectra"], self.data[idx]["mutant1"], self.data[idx]["mutant2"], self.data[idx][
            "mutant3"], self.data[idx]["mutant4"], self.data[idx]["sen_labels"]

    def __len__(self):  # What is the length of the dataset
        return len(self.data)


class MLP1(nn.Module):
    def __init__(self, num_input, num_hidden, num_class=32):
        super(MLP1, self).__init__()

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


class MLP2(nn.Module):
    def __init__(self, num_input, num_hidden, num_class=32):
        super(MLP2, self).__init__()

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


class MLP3(nn.Module):
    def __init__(self, num_input, num_hidden, num_class=32):
        super(MLP3, self).__init__()

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


class MLP4(nn.Module):
    def __init__(self, num_input, num_hidden, num_class=32):
        super(MLP4, self).__init__()

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


class MLP5(nn.Module):
    def __init__(self, num_input, num_hidden, num_class=32):
        super(MLP5, self).__init__()

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


class MLP11(nn.Module):
    def __init__(self, MLP2, MLP3, MLP4, MLP5, num_input, num_hidden, num_class=32):
        super().__init__()
        self.MLP1 = MLP2
        self.MLP2 = MLP3
        self.MLP3 = MLP4
        self.MLP4 = MLP5
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_class)
        self.Relu1 = nn.ReLU()
        self.Relu2 = nn.ReLU()

    def forward(self, M1, M2, M3, M4):
        m1 = self.MLP1.forward(M1)
        m2 = self.MLP2.forward(M2)
        m3 = self.MLP3.forward(M3)
        m4 = self.MLP4.forward(M4)

        x = torch.cat((m1, m2), dim=-1)
        x = torch.cat((x, m3), dim=-1)
        x = torch.cat((x, m4), dim=-1)

        x = self.linear1(x)
        x = self.Relu1(x)
        x = self.linear2(x)
        x = self.Relu2(x)
        return x


class MLP21(nn.Module):
    def __init__(self, MLP1, MLP11, num_input, num_hidden, num_class=1):
        super().__init__()
        self.MLP1 = MLP1
        self.MLP11 = MLP11
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_class)
        self.Relu1 = nn.ReLU()

    def forward(self, C, M1, M2, M3, M4):
        m1 = self.MLP1.forward(C)
        m2 = self.MLP11.forward(M1, M2, M3, M4)
        x = torch.cat((m1, m2), dim=-1)
        x = self.linear1(x)
        x = self.Relu1(x)
        x = self.linear2(x)
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
        for i, (cover, mutant1, mutant2, mutant3, mutant4, sen_label) in enumerate(iterator):
            cover = torch.tensor(cover).float().to(device)
            mutant1 = torch.tensor(mutant1).float().to(device)
            mutant2 = torch.tensor(mutant2).float().to(device)
            mutant3 = torch.tensor(mutant3).float().to(device)
            mutant4 = torch.tensor(mutant4).float().to(device)

            optimizer.zero_grad()

            output = model.forward(cover, mutant1, mutant2, mutant3, mutant4)

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
    c = MLP1(34, 64, 32).to(device)
    m1 = MLP2(35, 64, 32).to(device)
    m2 = MLP3(35, 64, 32).to(device)
    m3 = MLP4(35, 64, 32).to(device)
    m4 = MLP5(35, 64, 32).to(device)
    m5 = MLP11(m1, m2, m3, m4, 128, 64, 32).to(device)
    fmodel = MLP21(c, m5, 64, 128, 1).to(device)
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