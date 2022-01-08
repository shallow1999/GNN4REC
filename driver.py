import random
import time
from subprocess import Popen, DEVNULL
from sacred import Experiment
import os
import itertools
import multiprocessing
from multiprocessing import Process
from util.lt_util import exec_cmds, exec_cmd, parallel_exec_cmds

ex = Experiment()


@ex.config
def base_config():
    """
    tags是sacred中的一个默认参数字段，这里用来标注实验的类型，不同的实验类型对应不同的cmd执行方式
    debug表示调试代码，只执行一条cmd
    best表示最佳结果，执行多条cmd
    search表示搜索超参（grid search），用itertools.product生成并执行多条cmd
    analyze表示分析实验，例如消融实验，用itertools.product生成并执行多条cmd
    """
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
    # 要求配置文件必须以对应的模型名称开头
    model_name = config_name.split("_")[0]


@ex.automain
def main(gpus, max_proc_num, parallel_proc_num, wait_time, seed,
         tags, config_name, model_name, params, ex_name):
    # 通用的一些实验参数
    prefix = 'python main.py --name {} with "gpus={}" max_proc_num={} seed={}' \
             ' tags={} config_name={} model_name={}'.format(ex_name, gpus, max_proc_num,
                                                            seed, tags, config_name, model_name)
    # suffix = ">/dev/null 2>&1 &"
    suffix = ""

    # "params"用于填充特定的实验参数
    templete = '{} "params={}" {}'

    if tags == "debug":
        cmd = templete.format(prefix, params, suffix)
        exec_cmd(cmd)

    elif tags == "best":
        cmds = []
        for p in params.values():
            cmd = templete.format(prefix, p, suffix)
            cmds.append(cmd)

        random.shuffle(cmds)
        parallel_exec_cmds(parallel_proc_num=parallel_proc_num, wait_time=wait_time, cmds=cmds)

    elif tags in ["search", "analyze"]:
        keys = list(params.keys())
        values = list(params.values())
        p = {}
        cmds = []
        n = len(keys)
        ps = eval("itertools.product({})".format(", ".join(["values[{}]".format(i) for i in range(n)])))

        for t in ps:
            for i in range(n):
                p[keys[i]] = t[i]
            cmd = templete.format(prefix, p, suffix)
            cmds.append(cmd)

        # shuffle一下，既可以保证每个进程分到cmds的执行时间相近，也起到了random search的作用
        # ，根据经验，一般5%-10%左右的搜索空间就能找到近似最优参数了
        random.shuffle(cmds)
        parallel_exec_cmds(parallel_proc_num=parallel_proc_num, wait_time=wait_time, cmds=cmds)
