from dgl.nn.pytorch.conv import GraphConv
import torch.nn as nn
import torch.nn.functional as F
from net.base_net import BaseNet


class GCNNet(BaseNet):
    def __init__(self, n_user, n_item, in_dim, hid_dim, out_dim, num_layers=2, activation=F.relu):
        super(GCNNet, self).__init__(n_user, n_item, in_dim)
        self.gcn = nn.ModuleList()
        self.gcn.append(GraphConv(in_feats=in_dim, out_feats=hid_dim, activation=activation))

        for _ in range(num_layers - 2):
            self.gcn.append(GraphConv(in_feats=hid_dim, out_feats=hid_dim, activation=activation))

        self.gcn.append(GraphConv(in_feats=hid_dim, out_feats=out_dim, activation=None))

    def forward(self, graph, features):
        h = features

        for layer in self.gcn:
            h = layer(graph, h)

        return h

