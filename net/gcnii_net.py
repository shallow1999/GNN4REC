import math
import torch.nn as nn
import torch.nn.functional as F

from layer.gcnii_layer import GCNIILayer
from layer.gcnii_variant_layer import GCNIIVariantLayer
from util.lt_util import cal_gain
from net.base_net import BaseNet


class GCNIINet(BaseNet):
    def __init__(self, n_user, n_item, in_dim, hid_dim, out_dim, num_layers, bias=False,
                 activation=F.relu, graph_norm=True, dropout=0, alpha=0,
                 lamda=0, variant=False):
        super(GCNIINet, self).__init__(n_user, n_item, in_dim)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            beta = math.log(lamda / (i + 1) + 1)
            if variant:
                self.convs.append(GCNIIVariantLayer(hid_dim, hid_dim, bias, activation,
                                                    graph_norm, alpha, beta))
            else:
                self.convs.append(GCNIILayer(hid_dim, hid_dim, bias, activation,
                                             graph_norm, alpha, beta))
        self.fcs = nn.ModuleList()

        self.fcs.append(nn.Linear(in_dim, hid_dim))
        self.fcs.append(nn.Linear(hid_dim, out_dim))

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.fcs[0].weight, gain=gain)

        if self.fcs[0].bias is not None:
            nn.init.zeros_(self.fcs[0].bias)

        nn.init.xavier_uniform_(self.fcs[-1].weight)

        if self.fcs[-1].bias is not None:
            nn.init.zeros_(self.fcs[-1].bias)

    def forward(self, graph, features):
        h0 = F.dropout(features, self.dropout, self.training)
        h0 = self.activation(self.fcs[0](h0))

        h = h0
        for con in self.convs:
            h = F.dropout(h, self.dropout, self.training)
            h = con(graph, h, h0)

        h = F.dropout(h, self.dropout, self.training)
        h = self.fcs[-1](h)

        return h
