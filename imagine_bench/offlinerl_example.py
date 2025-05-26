import random
import argparse
from datetime import datetime

import torch
import numpy as np

import imagine_bench
from utils import LlataEncoderFactory, make_offlinerl_dataset
from gym import spaces
from algo.offlinerl.algo import algo_select
from evaluations import EvalCallBackFunction
from torch.utils.tensorboard import SummaryWriter

env = imagine_bench.make('Mujoco-v0', level='rephrase')
real_data, imaginary_rollout_rephrase = env.get_dataset(level="rephrase") 
dataset, data_num = make_offlinerl_dataset(real_data, imaginary_rollout_rephrase)

algo_args = {
    "algo_name": 'bc',
    "seed": 0,
    "obs_shape": env.observation_space.shape[0],
    "action_shape": env.action_space.shape[0],
    "device": "cuda:0",
    "exp_name": f"example",
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "max_action": float(spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32).high[0]),
    "task": 'mujoco'
}
writer  = SummaryWriter(f"logs")
algo_init_fn, algo_trainer_obj, algo_config = algo_select(algo_args)
algo_config['data_name'] = f"mujoco_rephrase"
algo_init = algo_init_fn(algo_config)
algo_trainer = algo_trainer_obj(algo_init, algo_config)
callback = EvalCallBackFunction()
callback.initialize({'train':env}, 10, writer)
algo_trainer.train(train_buffer=dataset, val_buffer=None, callback_fn=callback)