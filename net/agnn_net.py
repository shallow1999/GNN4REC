from dgl.nn.pytorch.conv import AGNNConv
import torch.nn as nn
from net.base_net import BaseNet


class AGNNNet(BaseNet):
    def __init__(self, n_user, n_item, in_dim, k=2, init_beta=1.0, learn_beta=False, allow_zero_in_degree=False):
        super(AGNNNet, self).__init__(n_user, n_item, in_dim)
        self.agnn = nn.ModuleList()
        self.agnn.append(AGNNConv(init_beta=init_beta, learn_beta=learn_beta, allow_zero_in_degree=allow_zero_in_degree))

        for _ in range(k - 2):
            self.agnn.append(AGNNConv(init_beta=init_beta, learn_beta=learn_beta, allow_zero_in_degree=allow_zero_in_degree))

        self.agnn.append(AGNNConv(init_beta=init_beta, learn_beta=learn_beta, allow_zero_in_degree=allow_zero_in_degree))

    def forward(self, graph, features):
        h = features

        for layer in self.agnn:
            h = layer(graph, h)

        return h

