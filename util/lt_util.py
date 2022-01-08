import os
import time
from pprint import pprint
from subprocess import Popen, DEVNULL
from multiprocessing import Process
import pynvml
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch as th
import random
from sacred import Experiment


def cal_gain(fun, param=None):
    """
    为Xavier初始化计算Gain
    :param fun: 非线性激活函数
    :param param: leaky_relu的negative_slope
    :return:
    """
    gain = 1
    if fun is F.sigmoid:
        gain = nn.init.calculate_gain('sigmoid')
    if fun is F.tanh:
        gain = nn.init.calculate_gain('tanh')
    if fun is F.relu:
        gain = nn.init.calculate_gain('relu')
    if fun is F.leaky_relu:
        gain = nn.init.calculate_gain('leaky_relu', param)
    return gain


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def log_metric(epoch, degree, **metric):
    """
    格式化输出epoch的结果
    :param epoch: 第几个epoch
    :param degree: 小数点后精度
    :param metric: 不定长字典参数，参考 https://blog.csdn.net/cadi2011/article/details/84871401
    :return:
    """
    info = "Epoch {:04d}".format(epoch)
    for key, value in metric.items():
        # 自带四舍五入功能，而且对于0.5也有处理，round对0.5就没有处理
        info += eval('" | {{}} {{:.{}f}}".format("{}", {})'.format(degree, key, value))
    print(info)


def rec_metric(ex: Experiment, epoch, degree, **metric):
    """

    :param ex: sacred的ex对象，这里用于将结果存进mongodb，key表示结果名称，value表示结果的值
    :param epoch: 第几个epoch
    :param degree: 小数点后精度
    :param metric: 不定长字典参数，参考 https://blog.csdn.net/cadi2011/article/details/84871401
    :return:
    """
    for key, value in metric.items():
        # 这里逢5不会进1
        value = round(value, degree)
        ex.log_scalar(key, value, epoch)


def log_rec_metric(ex: Experiment, epoch, degree, metric):
    """
    输出并记录结果
    :param ex:
    :param epoch:
    :param degree:
    :param metric:
    :return:
    """
    rec_metric(ex, epoch, degree, **metric)
    log_metric(epoch, degree, **metric)


def log_split(content="-" * 10, n=30):
    """
    输出分割线，content会居中显示，例如 --------------------------------run 1-------------------------------
    :param content:分割线中的内容
    :param n:分割线'-'左右各重复多少次
    :return:
    """
    print("\n{} {} {}\n".format("-" * n, content, "-" * n))


def generate_random_seeds(seed, nums):
    """
    由一个固定的随机种子产生一组固定的随机种子，是为了保证runs总的结果的复现性
    :param seed: 根随机种子
    :param nums: 产生的随机种子的数量
    :return:
    """
    random.seed(seed)
    # return [random.randint(1, 999999999) for _ in range(nums)]
    return [random.randint(0, 233333333) for _ in range(nums)]


def set_random_state(seed):
    """
    设置当前实验的随机种子，包括python，numpy，torch，保证结果可复现
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def get_gpu_proc_num(gpu=0):
    """
    获取当前GPU上的进程数，不过当加载数据时实际上获取不到该进程
    :param gpu: gpu编号
    :return:
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    process = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return len(process)


def get_free_gpu(gpus=[0], max_proc_num=2, max_wait=3600):
    """
    获取空闲的GPU集合
    :param gpus: 待检查的GPU集合
    :param max_proc_num: 一个GPU上最多可运行的进程数量
    :param max_wait: 当前没有空闲GPU时最大等待时间，超过该事件后抛出一个异常
    :return:
    """
    if th.cuda.is_available():
        waited = 0
        while True:
            # 优先使用最空闲的即运行进程最少的GPU
            for i in range(max_proc_num):
                for gpu in gpus:
                    if get_gpu_proc_num(gpu) == i:
                        return gpu
            time.sleep(10)
            waited += 10
            if waited > max_wait:
                raise Exception("There is no free gpu.")
    else:
        return "cpu"


def exec_cmd(cmd):
    """
    执行一条bash命令
    :param cmd: str格式的bash命令
    :return:
    """
    print("Running cmd: {}".format(cmd))
    proc = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    proc.wait()


# 等价于用&&拼接命令行，但是可以多开几个进程运行，从而实现并行化
def exec_cmds(cmds):
    """
    执行一组命令
    :param cmds:cmd list
    :return:
    """
    for cmd in cmds:
        exec_cmd(cmd)


def parallel_exec_cmds(parallel_proc_num, wait_time, cmds):
    """

    :param parallel_proc_num: 同时运行的进程的数量，每个进程预先均匀分配cmds，也就是说一个进程消耗完被分配的cmd子集后不会再去分担其他进程的
                              cmds，这会导致有些进程可能先消耗完，但是由于cmds命令经过了shuffle，总体上来说各个进程结束的时间还是比较
                              靠近的。这里也试过另外一种逻辑，即几个进程动态分配cmds，但是这样会导致都扎堆在一个GPU上，具体的原因过去太久了，
                              爷忘了
    :param wait_time: 由于加载数据时get_gpu_proc_num检测不到进程，因此这里需要设置一个合理的间隔时间，保证各个进程之间错开
    :param cmds: 总的cmds
    :return:
    """
    if parallel_proc_num > len(cmds):
        parallel_proc_num = len(cmds)

    procs = []
    # python list数组不存在越界问题，将来这里可以优化一下代码
    gap = int(len(cmds) / parallel_proc_num + 0.5)
    # 将总的cmds划分为一系列<=parallel_proc_num的子集
    for i in range(parallel_proc_num):
        start, end = i * gap, min(len(cmds), (i + 1) * gap)
        if start >= len(cmds):
            break
        batch_cmds = cmds[start:end]
        procs.append(Process(target=exec_cmds, args=(batch_cmds,)))
    for proc in procs:
        proc.start()
        time.sleep(wait_time)
    for proc in procs:
        proc.join()
