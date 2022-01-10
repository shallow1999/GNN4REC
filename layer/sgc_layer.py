import torch as th
from torch import nn
from torch.nn import Parameter
import dgl.function as fn
from torch.nn import functional as F


class SGCLayer(nn.Module):
    def __init__(self, k=2, neigh_aggr="degree", layer_aggr="mean"):
        """
        :param k: propagation stps
        :param layer_agrr: layer agrregate方式，有None
        """
        super(SGCLayer, self).__init__()
        self.k = k
        self.neigh_aggr = neigh_aggr
        self.layer_aggr = layer_aggr

    def forward(self, graph, features):
        """
        :param graph: 只包含边
        :param features: (M+N) X F的embedding，M是item的数量，N是user的数量
        :return: 传播后(M+N) X F的embedding
        """
        g = graph.local_var()
        h = features
        results = [features]

        # 这里保证度数都是1
        if self.neigh_aggr == "degree":
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)

        for _ in range(self.k):

            if self.neigh_aggr == "degree":
                h = h * norm
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
                h = g.ndata.pop('h')
                h = h * norm
            elif self.neigh_aggr == "mean":
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.mean('m', 'h'))
                h = g.ndata.pop('h')
            elif self.neigh_aggr == "pool":
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.max('m', 'h'))
                h = g.ndata.pop('h')

            results.append(h)

        if self.layer_aggr == "mean":
            H = th.stack(results, dim=1)
            H = th.mean(H, dim=1)
        elif self.layer_aggr == "1/k":
            H = results[0]
            for i in range(1, len(results)):
                emb = results[i] / (i + 1)
                H = H + emb
        elif self.layer_aggr == "cat":
            H = th.cat(results, dim=1)
        elif self.layer_aggr == "none":
            return results[-1]

        return H