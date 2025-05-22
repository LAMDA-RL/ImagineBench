import os
import random
import argparse
from datetime import datetime

import torch
import numpy as np

import imagine_bench
import algo.d3rlpy as d3rlpy
from evaluations import CallBack
from algo.d3rlpy.logging import TensorboardAdapterFactory
from utils import LlataEncoderFactory, make_d3rlpy_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Mujoco-v0", help="The name of Env.")
    parser.add_argument("--ds_type", type=str, default="train", choices=['train', 'rephrase', 'easy', 'hard'], help="The type of offlineRL dataset.")
    parser.add_argument("--algo", type=str, default="bc", choices=['bc', 'cql', 'bcq', 'td3+bc'], help="The name of offlineRL agent.")
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
    discrete_action = True if env_name in ['BabyAI-v0', 'Ball-v0'] else False

    if level == "train":
        env = imagine_bench.make(env_name, level='real')
    else:
        env = imagine_bench.make(env_name, level=level)
    if level == "train":
        real_data, imaginary_rollout = env.get_dataset(level="rephrase") 
        dataset = make_d3rlpy_dataset(real_data, None)
    else:
        real_data, imaginary_rollout = env.get_dataset(level=level) 
        dataset = make_d3rlpy_dataset(real_data, imaginary_rollout)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if not discrete_action:
        if args.algo == 'bc':
            agent = d3rlpy.algos.BCConfig(
                encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            ).create(device=args.device)
        elif args.algo == 'cql':
            agent = d3rlpy.algos.CQLConfig(
                actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
                critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            ).create(device=args.device)
        elif args.algo == 'bcq':
            agent = d3rlpy.algos.BCQConfig(
                actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
                critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            ).create(device=args.device)
        elif args.algo == 'td3+bc':
            alg_hyper_list = [
            ]
            agent = d3rlpy.algos.TD3PlusBCConfig(
                actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
                critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            ).create(device=args.device)
        else:
            raise NotImplementedError
    else:
        f_size = 64 if env_name == 'BabyAI-v0' else 256
        if args.algo == 'bc':
            agent = d3rlpy.algos.DiscreteBCConfig(
                encoder_factory=LlataEncoderFactory(feature_size=f_size, hidden_size=f_size),
            ).create(device=args.device)
        elif args.algo == 'cql':
            agent = d3rlpy.algos.DiscreteCQLConfig(
                encoder_factory=LlataEncoderFactory(feature_size=f_size, hidden_size=f_size),
                alpha=10.0,
            ).create(device=args.device)
        elif args.algo == 'bcq':
            agent = d3rlpy.algos.DiscreteBCQConfig(
                encoder_factory=LlataEncoderFactory(feature_size=f_size, hidden_size=f_size),
            ).create(device=args.device)
        elif args.algo == 'sac':
            agent = d3rlpy.algos.DiscreteSACConfig(
                actor_encoder_factory=LlataEncoderFactory(feature_size=f_size, hidden_size=f_size),
                critic_encoder_factory=LlataEncoderFactory(feature_size=f_size, hidden_size=f_size),
            ).create(device=args.device)
        else:
            raise NotImplementedError
    
    kwargs = vars(args)    
    exp_name = f'{exp_name}_{"i-" if level != "train" else ""}{kwargs["ds_type"]}'
    exp_name_temp = f'{exp_name}_{kwargs["algo"]}_seed{kwargs["seed"]}'
    exp_name = f'{exp_name_temp}_{datetime.now().strftime("%m-%d_%H-%M-%S")}'

    real_env = imagine_bench.make(env_name, level='real')
    rephrase_env = imagine_bench.make(env_name, level='rephrase')
    easy_env = imagine_bench.make(env_name, level='easy')
    hard_env = imagine_bench.make(env_name, level='hard')
    if level == "train":
        env_dict = {"train": real_env, 
                    'rephrase': rephrase_env,
                    'easy': easy_env,
                    'hard': hard_env}
    elif level == 'rephrase':
        env_dict = {"train": real_env, 
                    'rephrase': rephrase_env}
    elif level == 'easy':
        env_dict = {'easy': easy_env}
    elif level == 'hard':
        env_dict = {'hard': hard_env}
    callback = CallBack()
    callback.add_eval_env(env_dict=env_dict, eval_num=args.eval_episodes)

    # offline training
    agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name=exp_name,
        with_timestamp=False,
        logger_adapter=TensorboardAdapterFactory(root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        epoch_callback=callback.EvalCallback,
    )

    print("===> offline training finished")
