import os
import time
import random
import numpy as np
from loguru import logger

from offlinerl.utils.logger import log_path
from offlinerl.utils.io import create_dir, download_helper, read_json

from offlinerl.data.neorl import load_neorl_buffer

dataset_dir = os.path.join(log_path(),"./offlinerl_datasets")
create_dir(dataset_dir)

def load_data_from_neorl2_util(task):

    import neorl2
    import gymnasium as gym

    env = neorl2.make(task)
    if 'fusion' in task.lower():
        train_data, val_data = env.get_dataset(traj_num=20)
    else:
        train_data, val_data = env.get_dataset()

    return train_data, val_data

def load_data_from_neorl2(task):
    train_data, val_data = load_data_from_neorl2_util(task)
    train_buffer = load_neorl_buffer({
        'obs': train_data["obs"].astype(np.float32),
        'action': train_data["action"].astype(np.float32),
        'next_obs': train_data["next_obs"].astype(np.float32),
        'reward': train_data["reward"].astype(np.float32).reshape(-1, 1),
        'done': np.bool_(train_data["done"]).reshape(-1, 1),
    })
    
    val_buffer = load_neorl_buffer({
        'obs': val_data["obs"].astype(np.float32),
        'action': val_data["action"].astype(np.float32),
        'next_obs': val_data["next_obs"].astype(np.float32),
        'reward': val_data["reward"].astype(np.float32).reshape(-1, 1),
        'done': np.bool_(val_data["done"]).reshape(-1, 1),
    })

    return train_buffer, val_buffer

def load_data_from_neorl(task, task_data_type = "low", task_train_num = 99):
    try:
        import neorl
        env = neorl.make(task)
        train_data, val_data = env.get_dataset(data_type = task_data_type, train_num = task_train_num)
        train_buffer, val_buffer = load_neorl_buffer(train_data), load_neorl_buffer(val_data)
        logger.info(f"Load task data from neorl. -> {task}")
    except:
        train_buffer, val_buffer = load_data_from_neorl2(task)
        logger.info(f"Load task data from neorl2. -> {task}")
    return train_buffer, val_buffer