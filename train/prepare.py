import torch.nn as nn
import torch as th

from net.appnp_net import APPNPNet
from net.dagnn_net import DAGNNNet
from net.gat_net import GATNet
from net.gcn_net import GCNNet
from net.gcnii_net import GCNIINet
from net.sage_net import SAGENet
from train.dataset import Dataset
from net.sgc_net import SGCNet


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
        graph = dataset.get_dgl_graph().to(self.device)
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
                aggr=self.params["aggr"]
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
                dropout=self.params["dropout"]
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
                attn_drop=self.params["attn_drop"]
            )
        else:
            raise NotImplementedError(self.model_name)

        model = model.to(self.device)

        optimizer = th.optim.Adam(model.parameters(), lr=self.params["lr"])
        print(model)
        for name, param in model.named_parameters(recurse=True):
            print(f"name:{name}, param:{param}")

        self.model, self.optimizer = model, optimizer

        return model, optimizer
