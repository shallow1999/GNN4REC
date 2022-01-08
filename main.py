from pprint import pprint

from sacred import Experiment
from sacred.observers import MongoObserver
from train.prepare import Prepare
from train.train import train
from util.lt_util import generate_random_seeds, set_random_state, get_free_gpu, log_split, log_rec_metric
import torch as th
from train.test import test

# TODO Sacred固定代码
ex = Experiment()
# 配置mongodb地址
ex.observers.append(MongoObserver(url='10.192.9.122:7000',
                                  db_name='WJY'))


# TODO Sacred固定代码
@ex.config
def base_config():
    tags = "debug"
    config_name = None

    if not config_name:
        raise ValueError("Please input config_name")

    if tags == "debug":
        ex.add_config('config/base_config/{}.json'.format(config_name))
    elif tags == "best":
        ex.add_config("config/best_config/{}.json".format(config_name))
    elif tags == "search":
        ex.add_config("config/search_config/{}.json".format(config_name))
    elif tags == "analyze":
        ex.add_config("config/analyze_config/{}.json".format(config_name))
    else:
        raise Exception("There is no {}".format(tags))

    ex_name = config_name
    model_name = config_name.split("_")[0]


@ex.automain
def main(gpus, max_proc_num, seed, model_name, params):
    print("model: {}".format(model_name))
    pprint(params)  # TODO 可选代码，输出参数

    device = get_free_gpu(gpus, max_proc_num)  # TODO 固定代码

    prepare = Prepare(device, params, model_name)
    prepare.prepare_data()

    random_seeds = generate_random_seeds(seed, params["num_runs"])  # TODO 固定代码
    for run in range(params["num_runs"]):
        # 一定要放在最前面，确保接下来的所有操作都是可复现的
        # TODO 但是如果下面存在必须随机的操作就会有问题，比如shuffle数据
        # set_random_state(random_seeds[run])  # TODO 可选代码

        prepare.prepare_model()
        n_log_run = 5
        # 只记录前几个runs的logs
        if run < n_log_run:
            log_split(" {}th run ".format(run + 1))  # TODO 可选代码，输出分割log

        counter = 0
        best_score = 0

        for epoch in range(1, params['num_epochs'] + 1):

            avg_loss = train(prepare, params["train_batch_size"], params["emb_regular"])

            log_rec_metric(ex, epoch, 4, {"avg_loss": avg_loss})  # TODO 可选代码，记录每个epoch的结果数据

            if epoch % 10 == 0:
                print("Test")
                recall, precis, ndcg = test(prepare, params["test_batch_size"])
                metric = {"precis": precis,
                          "recall": recall,
                          "ndcg": ndcg
                          }

                log_rec_metric(ex, epoch, 4, metric)  # TODO 可选代码，记录每个epoch的结果数据

                # 临时的early stopping，后面有时间加上验证集，现在就算了，不想搞了
                if recall > best_score:
                    best_score = recall
                    counter = 0
                else:
                    counter += 1
                if counter >= 10:
                    break
