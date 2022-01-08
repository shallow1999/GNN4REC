import torch.nn as nn
import torch.nn.functional as F
from util.lt_util import cal_gain
import torch as th


class BaseNet(nn.Module):
    def __init__(self, n_user, n_item, emb_dim):
        super(BaseNet, self).__init__()
        self.n_user, self.n_item = n_user, n_item

        self.emb_users_ini = nn.Embedding(num_embeddings=n_user, embedding_dim=emb_dim)
        self.emb_items_ini = nn.Embedding(num_embeddings=n_item, embedding_dim=emb_dim)
        self.init_embedding()

    def init_embedding(self):
        nn.init.normal_(self.emb_users_ini.weight, std=0.1)
        nn.init.normal_(self.emb_items_ini.weight, std=0.1)

    def get_features(self):
        emb_features = th.cat([self.emb_users_ini.weight, self.emb_items_ini.weight])
        return emb_features

    def split_emb_out(self, emb_out):
        emb_users_out, emb_items_out = th.split(emb_out, [self.n_user, self.n_item])
        return emb_users_out, emb_items_out

    def compute_rating(self, part_users, emb_users_out, emb_items_out):
        emb_part_users_out = emb_users_out[part_users]
        rating = F.sigmoid(th.matmul(emb_part_users_out, emb_items_out.t()))
        return rating

    def get_emb_out(self, part_users, pos_items, neg_items, emb_users_out, emb_items_out):
        emb_part_users_out = emb_users_out[part_users]
        emb_pos_out = emb_items_out[pos_items]
        emb_neg_out = emb_items_out[neg_items]
        return emb_part_users_out, emb_pos_out, emb_neg_out

    def get_emb_ini(self, part_users, pos_items, neg_items):
        # 后续优化一下long()
        emb_part_users_ini = self.emb_users_ini(part_users)
        emb_pos_ini = self.emb_items_ini(pos_items)
        emb_neg_ini = self.emb_items_ini(neg_items)
        return emb_part_users_ini, emb_pos_ini, emb_neg_ini


