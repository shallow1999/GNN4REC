import torch.nn as nn
import torch as th

from net.agnn_net import AGNNNet
from net.appnp_net import APPNPNet
from net.dagnn_net import DAGNNNet
from net.gat_net import GATNet
from net.gcn_net import GCNNet
from net.gcnii_net import GCNIINet
from net.sage_net import SAGENet
from train.dataset import Dataset
from net.sgc_net import SGCNet
from util.lt_util import log_split


class Prepare(object):
    def __init__(self, device, params, model_name):
        self.device = device
        self.params = params
        self.model_name = model_name

        self.graph, self.test_dict, self.dataset = None, None, None
        self.n_users, self.n_items = None, None
        self.model, self.optimizer = None, None

    def prepare_data(self):
        dataset = Dataset(dataset=self.params["dataset"])

        graph = dataset.get_dgl_graph()
        log_split('graph info')
        print(graph)
        # 加上自环
        graph = graph.add_self_loop()
        graph = graph.to(self.device)

        test_dict = dataset.test_dict

        self.graph, self.test_dict, self.dataset = graph, test_dict, dataset
        self.n_users, self.n_items = dataset.n_users, dataset.n_items

        return graph, test_dict, dataset

    def prepare_model(self):
        if self.model_name == "sgc":
            model = SGCNet(
                n_user=self.n_users,
                n_item=self.n_items,
                in_dim=self.params["emb_dim"],
                k=self.params["k"],
                neigh_aggr=self.params["neigh_aggr"],
                layer_aggr=self.params["layer_aggr"]
            )
        elif self.model_name == "dagnn":
            model = DAGNNNet(
                n_user=self.n_users,
                n_item=self.n_items,
                in_dim=self.params["emb_dim"],
                hid_dim=self.params["hid_dim"],
                out_dim=self.params["emb_dim"],
                k=self.params["k"],
                dropout=self.params["dropout"]
            )
        elif self.model_name == "appnp":
            model = APPNPNet(
                n_user=self.n_users,
                n_item=self.n_items,
                in_dim=self.params["emb_dim"],
                hid_dim=self.params["hid_dim"],
                out_dim=self.params["out_dim"],
                k=self.params["k"],
                alpha=self.params["alpha"],
                dropout=self.params["dropout"],
                edge_drop=self.params["edge_drop"]
            )
        elif self.model_name == "gcn":
            model = GCNNet(
                n_user=self.n_users,
                n_item=self.n_items,
                in_dim=self.params["emb_dim"],
                hid_dim=self.params["hid_dim"],
                out_dim=self.params["out_dim"],
                num_layers=self.params["num_layers"]
            )
        elif self.model_name == 'gcnii':
            model = GCNIINet(
                n_user=self.n_users,
                n_item=self.n_items,
                in_dim=self.params["emb_dim"],
                hid_dim=self.params["hid_dim"],
                out_dim=self.params["out_dim"],
                num_layers=self.params["num_layers"],
                dropout=self.params["dropout"],
                alpha=self.params["alpha"],
                lamda=self.params["lamda"]
            )
        elif self.model_name == 'sage':
            model = SAGENet(
                n_user=self.n_users,
                n_item=self.n_items,
                in_dim=self.params["emb_dim"],
                hid_dim=self.params["hid_dim"],
                out_dim=self.params["out_dim"],
                num_layers=self.params["num_layers"],
                aggregator_type=self.params["aggregator_type"]
            )
        elif self.model_name == 'gat':
            model = GATNet(
                n_user=self.n_users,
                n_item=self.n_items,
                in_dim=self.params["emb_dim"],
                hid_dim=self.params["hid_dim"],
                out_dim=self.params["out_dim"],
                num_heads=self.params["num_heads"],
                num_layers=self.params["num_layers"],
                feat_drop=self.params["feat_drop"],
                attn_drop=self.params["attn_drop"],
                residual=self.params["residual"]
            )
        elif self.model_name == 'agnn':
            model = AGNNNet(
                n_user=self.n_users,
                n_item=self.n_items,
                in_dim=self.params["emb_dim"],
                k=self.params["k"]
            )
        else:
            raise NotImplementedError(self.model_name)

        model = model.to(self.device)
        log_split(f'{self.model_name} info')
        print(model)
        optimizer = th.optim.Adam(model.parameters(), lr=self.params["lr"])

        self.model, self.optimizer = model, optimizer

        return model, optimizer
