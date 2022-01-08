import multiprocessing

from train.loss import BPRLoss
from util.data_util import uniform_sample
import torch as th
from util.data_util import shuffle, minibatch
from train.metric import recall_and_precis_atk, ndgc_atk
import numpy as np
from net.base_net import BaseNet


# TODO 后面加个full batch GPU Torch版本的，现在评估指标是在CPU上用numpy计算的，效率太低
def test(prepare, test_batch_size):
    graph, test_dict, dataset = prepare.graph, prepare.test_dict, prepare.dataset
    model, optimizer = prepare.model, prepare.optimizer
    topk = prepare.params["topk"]

    multicore = prepare.params["multicore"]
    if multicore > 0:
        pool = multiprocessing.Pool(multicore)

    batches_labels, batches_preds = [], []
    model.eval()
    with th.no_grad():
        test_users = list(test_dict.keys())
        recall, precis, ndcg = 0., 0., 0.

        # 前向传播一次即可
        emb_features = model.get_features()
        emb_out = model(graph, emb_features)
        emb_users_out, emb_items_out = model.split_emb_out(emb_out)

        for batch_users in minibatch(test_batch_size, test_users):
            all_pos = dataset.get_user_pos_items(batch_users)
            batch_labels = [test_dict[u] for u in batch_users]
            batch_users = th.Tensor(batch_users).long().to(prepare.device)

            # emb_out = model(graph, emb_features)
            # emb_users_out, emb_items_out = split_emb_out(prepare.n_users, prepare.n_items, emb_out)

            rating = model.compute_rating(batch_users, emb_users_out, emb_items_out)

            exc_idxs, exc_items = [], []
            for i, items in enumerate(all_pos):
                exc_idxs.extend([i] * len(items))
                exc_items.extend(items)
            rating[exc_idxs, exc_items] = -(1<<10)
            _, batch_preds = th.topk(rating, k=topk)
            batch_preds = batch_preds.cpu()

            batches_labels.append(batch_labels)
            batches_preds.append(batch_preds)
    labels_and_preds = zip(batches_labels, batches_preds, [topk for _ in range(len(batches_labels))])
    if multicore > 0:
        results = pool.map(test_one_batch, labels_and_preds)
    else:
        results = [test_one_batch(x) for x in labels_and_preds]

    recall, precis, ndcg = 0., 0., 0.
    for result in results:
        recall += result["recall"]
        precis += result["precis"]
        ndcg += result["ndcg"]
    recall /= len(test_users)
    precis /= len(test_users)
    ndcg /= len(test_users)

    recall = round(recall, 5)
    precis = round(precis, 5)
    ndcg = round(ndcg, 5)

    if multicore > 0:
        pool.close()

    return recall, precis, ndcg


def test_one_batch(labels_and_preds):
    batch_labels = labels_and_preds[0]
    batch_preds = labels_and_preds[1]
    topk = labels_and_preds[2]
    batch_preds = tran_one_zero(batch_labels, batch_preds)
    precis, recall = recall_and_precis_atk(batch_labels, batch_preds, topk)
    ndcg = ndgc_atk(batch_labels, batch_preds, topk)

    result = {
        "precis": precis,
        "recall": recall,
        "ndcg": ndcg
    }
    return result


def tran_one_zero(labels, pred):
    result = []
    for i in range(len(labels)):
        tran = list(map(lambda x: x in labels[i], pred[i]))
        tran = np.array(tran).astype("float")
        result.append(tran)
    return np.array(result).astype("float")