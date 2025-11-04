import os
import random
import argparse
from datetime import datetime

import torch
import numpy as np

import imagine_bench
from utils import LlataEncoderFactory, make_d3rlpy_dataset, make_offlinerl_dataset
from gym import spaces
from algo.offlinerl.algo import algo_select
from evaluations import EvalCallBackFunction
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Mujoco-v0", help="The name of Env.")
    parser.add_argument("--ds_type", type=str, default="train", choices=['train', 'rephrase', 'easy', 'hard'], help="The type of offlineRL dataset.")
    parser.add_argument("--algo", type=str, default="bc", choices=['bc', 'cql', 'bcq', 'td3+bc', 'prdc', 'combo'], help="The name of offlineRL agent.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device for offlineRL training.")
    # offlineRL algorithm hyperparameters
    parser.add_argument("--seed", type=int, default=7, help="Seed.")

    parser.add_argument("--eval_episodes", type=int, default=10)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    env_name = args.env
    exp_name = args.env[:-3]
    level = args.ds_type
    if level == "train":
        env = imagine_bench.make(env_name, level='real')
    else:
        env = imagine_bench.make(env_name, level=level)
    if level == "train":
        real_data, imaginary_rollout = env.get_dataset(level="rephrase") 
        dataset, data_num = make_offlinerl_dataset(real_data, None)
    else:
        real_data, imaginary_rollout = env.get_dataset(level=level) 
        dataset, data_num = make_offlinerl_dataset(real_data, imaginary_rollout)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    algo_args = {
        "algo_name": args.algo,
        "seed": args.seed,
        "obs_shape": env.observation_space.shape[0],
        "action_shape": env.action_space.shape[0],
        "device": args.device,
        "exp_name": f"{args.env}_{args.ds_type}_{args.algo}_{args.seed}",
        "state_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "max_action": float(spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32).high[0]),
        "task": args.env[:-3].lower()
        # "obs_space":obs_space,
        # "action_space":action_space
    }
    writer  = SummaryWriter(f"logs/{args.algo}+dataset={args.ds_type}+data_num={data_num}+seed={args.seed}")
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(algo_args)
    algo_config['data_name'] = f"{args.env}_{args.ds_type}"
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    callback = EvalCallBackFunction()
    real_env = imagine_bench.make(env_name, level='real')
    rephrase_env = imagine_bench.make(env_name, level='rephrase')
    easy_env = imagine_bench.make(env_name, level='easy')
    hard_env = imagine_bench.make(env_name, level='hard')
    if args.ds_type == 'train':
        eval_dict = {'train': real_env, 
                     'rephrase': rephrase_env, 
                     'easy':easy_env, 
                     'hard':hard_env}
    elif args.ds_type == 'rephrase':
        eval_dict = {'train':real_env, 
                     'rephrase':rephrase_env}
    elif args.ds_type == 'easy':
        eval_dict = {'train':real_env, 
                     'easy':easy_env}
    elif args.ds_type == 'hard':
        eval_dict = {'train':real_env, 
                     'hard':hard_env}
    callback.initialize(eval_dict, args.eval_episodes, writer)
    algo_trainer.train(train_buffer=dataset, val_buffer=None, callback_fn=callback)