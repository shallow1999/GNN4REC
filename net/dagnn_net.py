from layer.dagnn_layer import DAGNNLayer
import torch.nn as nn
import torch.nn.functional as F
from util.lt_util import cal_gain
from net.base_net import BaseNet

class DAGNNNet(BaseNet):
    def __init__(self, n_user, n_item, in_dim, hid_dim, out_dim, k, bias=True, activation=F.relu, dropout=0):
        super(DAGNNNet, self).__init__(n_user, n_item, in_dim)
        self.linear1 = nn.Linear(in_dim, hid_dim, bias)
        self.linear2 = nn.Linear(hid_dim, out_dim, bias)

        self.dagnn = DAGNNLayer(out_dim, k)
        self.activation = activation
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)

        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight, gain=gain)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, graph, features):
        h = F.dropout(features, self.dropout, training=self.training)
        h = self.activation(self.linear1(h))

        h = F.dropout(h, self.dropout, training=self.training)
        h = self.linear2(h)

        h = self.dagnn(graph, h)

        return h