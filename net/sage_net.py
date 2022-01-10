from dgl.nn.pytorch.conv import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from net.base_net import BaseNet


class SAGENet(BaseNet):
    def __init__(self, n_user, n_item, in_dim, hid_dim, out_dim, num_layers, aggregator_type, feat_drop=0.0, bias=True,
                 norm=None, activation=F.relu):
        super(SAGENet, self).__init__(n_user, n_item, in_dim)
        self.sage = nn.ModuleList()

        self.sage.append(SAGEConv(in_feats=in_dim, out_feats=hid_dim, aggregator_type=aggregator_type,
                                  feat_drop=feat_drop, bias=bias, norm=norm, activation=activation))

        for _ in range(num_layers - 2):
            self.sage.append(SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type=aggregator_type,
                                      feat_drop=feat_drop, bias=bias, norm=norm, activation=activation))

        self.sage.append(SAGEConv(in_feats=hid_dim, out_feats=out_dim, aggregator_type=aggregator_type,
                                  feat_drop=feat_drop, bias=bias, norm=norm, activation=None))

    def forward(self, graph, features):
        h = features

        for layer in self.sage:
            h = layer(graph, h)

        return h