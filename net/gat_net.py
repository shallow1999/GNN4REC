from dgl.nn.pytorch.conv import GATConv
import torch.nn as nn
import torch.nn.functional as F
from net.base_net import BaseNet


class GATNet(BaseNet):
    def __init__(self, n_user, n_item, in_dim, hid_dim, out_dim, num_heads, num_layers, feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=False, activation=F.elu, allow_zero_in_degree=False):
        super(GATNet, self).__init__(n_user, n_item, in_dim)

        self.gat = nn.ModuleList()

        self.gat.append(GATConv(in_feats=in_dim, out_feats=hid_dim // num_heads, num_heads=num_heads,
                                feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                                residual=residual, activation=activation, allow_zero_in_degree=allow_zero_in_degree))

        for _ in range(num_layers - 2):
            self.gat.append(
                GATConv(in_feats=hid_dim, out_feats=hid_dim // num_heads, num_heads=num_heads, feat_drop=feat_drop,
                        activation=activation, allow_zero_in_degree=allow_zero_in_degree))

        self.gat.append(
            GATConv(in_feats=hid_dim, out_feats=out_dim // num_heads, num_heads=num_heads, feat_drop=feat_drop,
                    attn_drop=attn_drop, negative_slope=negative_slope, residual=residual,
                    activation=None, allow_zero_in_degree=allow_zero_in_degree))

    def forward(self, graph, features):
        h = features

        for layer in self.gat:
            h = layer(graph, h).flatten(1)

        return h
