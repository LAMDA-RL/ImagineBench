import os
import uuid
import random


import torch
import numpy as np
from aim import Run
from loguru import logger

from offlinerl.utils.logger import log_path


def setup_seed(seed=1024):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
def select_free_cuda():
    # 获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("No GPU available.")
        return None

    # 遍历所有 GPU，选择利用率最低的 GPU
    min_memory_usage = float('inf')
    selected_gpu_id = None

    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        gpu_memory_usage = torch.cuda.max_memory_allocated() / 1024**3  # in GB
        # 选择利用率最低的 GPU
        if gpu_memory_usage < min_memory_usage:
            min_memory_usage = gpu_memory_usage
            selected_gpu_id = gpu_id

    return selected_gpu_id

def set_free_device_fn():
    device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'

    return device


def init_exp_run(repo=None, experiment_name=None, flush_frequency=1):
    if repo is None:
        repo = os.path.join(log_path(),"./.aim")
        if not os.path.exists(repo):
            print(f'=====repo:{repo}')
            logger.info('{} dir is not exist, create {}',repo, repo)
            os.system(str("cd " + os.path.join(repo,"../") + "&& aim init"))
    else:
        repo = os.path.join(repo,"./.aim")
        if not os.path.exists(repo):
            print(f'=====repo:{repo}')
            logger.info('{} dir is not exist, create {}',repo, repo)
            os.system(str("cd " + os.path.join(repo,"../") + "&& aim init"))
    run = Run(
        repo=repo,
        experiment=experiment_name
    )
        
    return run