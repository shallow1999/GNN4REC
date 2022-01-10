from layer.sgc_layer import SGCLayer
import torch.nn as nn
from net.base_net import BaseNet


class SGCNet(BaseNet):
    def __init__(self, n_user, n_item, in_dim, k=2, neigh_aggr="degree", layer_aggr="none"):
        super(SGCNet, self).__init__(n_user, n_item, in_dim)
        self.rsgc = SGCLayer(k, neigh_aggr, layer_aggr)

    def forward(self, graph, features):
        h = self.rsgc(graph, features)
        return h
