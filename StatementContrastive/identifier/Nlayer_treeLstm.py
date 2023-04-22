"""
PyTorch Child-Sum Tree LSTM model

See Tai et al. 2015 https://arxiv.org/abs/1503.00075 for model description.
"""

import torch
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

