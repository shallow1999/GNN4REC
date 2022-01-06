import random
import numpy as np
import time
import os
import time
from subprocess import Popen, DEVNULL
from multiprocessing import Process
import numpy as np
import random
import pynvml
from train.loss import BPRLoss
from util.data_util import uniform_sample
import torch as th
from util.data_util import shuffle, minibatch
from util.emb_util import get_emb_out, split_emb_out, get_emb_ini
from sacred import Experiment


def train(prepare, train_batch_size, emb_regular):
    graph, test_dict, dataset = prepare.graph, prepare.test_dict, prepare.dataset
    emb_users_ini, emb_items_ini, emb_features = prepare.emb_users_ini, prepare.emb_items_ini, prepare.emb_features
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
        emb_out = model(graph, emb_features)
        emb_users_out, emb_items_out = split_emb_out(dataset.n_users, dataset.n_items, emb_out)
        emb_part_users_out, emb_pos_out, emb_neg_out = get_emb_out(batch_users, batch_pos, batch_neg,
                                                                   emb_users_out, emb_items_out)
        emb_part_users_ini, emb_pos_ini, emb_neg_ini = get_emb_ini(batch_users, batch_pos, batch_neg,
                                                                   emb_users_ini, emb_items_ini)

        loss = BPRLoss(emb_regular, emb_part_users_out, emb_pos_out, emb_neg_out,
                       emb_part_users_ini, emb_pos_ini, emb_neg_ini)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss
    avg_loss /= n_batch
    return avg_loss.item()