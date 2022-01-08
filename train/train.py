from train.loss import BPRLoss
from util.data_util import uniform_sample
import torch as th
from util.data_util import shuffle, minibatch
from sacred import Experiment
from util.lt_util import log_split


def train(prepare, train_batch_size, emb_regular):
    graph, test_dict, dataset = prepare.graph, prepare.test_dict, prepare.dataset
    model, optimizer = prepare.model, prepare.optimizer

    samples = uniform_sample(dataset)
    # 后期把这里参数优化一下
    sample_users = th.Tensor(samples[:, 0]).long().to(prepare.device)
    sample_pos = th.Tensor(samples[:, 1]).long().to(prepare.device)
    sample_neg = th.Tensor(samples[:, 2]).long().to(prepare.device)
    # print(sample_users)
    sample_users, sample_pos, sample_neg = shuffle(sample_users, sample_pos, sample_neg)
    # print(sample_users)
    n_batch = len(sample_users) // train_batch_size + 1
    avg_loss = 0.

    model.train()

    for (i, (batch_users, batch_pos, batch_neg)) in enumerate(minibatch(train_batch_size, sample_users,
                                                                        sample_pos, sample_neg)):

        emb_features = model.get_features()
        emb_out = model(graph, emb_features)

        # log_split(f"emb_features {i}", 30)
        # print(emb_features[batch_users[0]][:10])
        # print(emb_features[1][:10])
        #
        # log_split(f"emb_users_ini {i}", 30)
        # print(model.emb_users_ini.weight[batch_users[0]][:10])
        # print(model.emb_users_ini.weight[1][:10])

        emb_users_out, emb_items_out = model.split_emb_out(emb_out)
        emb_part_users_out, emb_pos_out, emb_neg_out = model.get_emb_out(batch_users, batch_pos, batch_neg,
                                                                         emb_users_out, emb_items_out)
        emb_part_users_ini, emb_pos_ini, emb_neg_ini = model.get_emb_ini(batch_users, batch_pos, batch_neg)

        loss = BPRLoss(emb_regular, emb_part_users_out, emb_pos_out, emb_neg_out,
                       emb_part_users_ini, emb_pos_ini, emb_neg_ini)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss

        # log_split(f"emb_features_2 {i}", 30)
        # print(emb_features[batch_users[0]][:10])
        # print(emb_features[1][:10])
        #
        # log_split(f"emb_users_ini_2 {i}", 30)
        # print(model.emb_users_ini.weight[batch_users[0]][:10])
        # print(model.emb_users_ini.weight[1][:10])

    avg_loss /= n_batch
    return avg_loss.item()
