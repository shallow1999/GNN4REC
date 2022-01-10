import torch.nn as nn
from dgl.nn.pytorch import APPNPConv
import torch.nn.functional as F
from util.lt_util import cal_gain
from net.base_net import BaseNet


class APPNPNet(BaseNet):
    def __init__(self, n_user, n_item, in_dim, hid_dim, out_dim, k, alpha, dropout=0, edge_drop=0):
        super(APPNPNet, self).__init__(n_user, n_item, in_dim)
        # self.linear1 = nn.Linear(in_dim, hid_dim)
        # self.linear2 = nn.Linear(hid_dim, out_dim)
        # self.activation = F.relu
        # self.dropout = dropout
        self.rappnp = APPNPConv(k, alpha, edge_drop=edge_drop)
        # self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)

        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, graph, features):
        # h = F.dropout(features, self.dropout, training=self.training)
        # h = self.activation(self.linear1(h))
        # h = F.dropout(h, self.dropout, training=self.training)
        # h = self.linear2(h)
        h = features
        h = self.rappnp(graph, h)

        return h