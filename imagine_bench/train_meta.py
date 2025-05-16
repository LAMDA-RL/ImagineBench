import h5py
import torch
import random
import argparse
import dataclasses
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import Any, Dict, List, Union, Sequence
from algo.d3rlpy.dataset.components import Episode
from algo.d3rlpy.models.encoders import EncoderFactory, register_encoder_factory
from algo.d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from algo.d3rlpy.dataset.types import Observation, ObservationSequence
from algo.d3rlpy.dataset.mini_batch import TransitionMiniBatch, stack_observations, cast_recursively
from algo.d3rlpy.dataset.transition_pickers import Transition, TransitionPickerProtocol, _validate_index, retrieve_observation, create_zero_observation

import os
import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

from algo import d3rlpy
from algo.d3rlpy.logging import TensorboardAdapterFactory
from algo.d3rlpy.algos.qlearning import QLearningAlgoBase

import imagine_bench
from imagine_bench.utils import LlataEncoderFactory, make_d3rlpy_dataset
from imagine_bench.envs import DATASET_PATH

from imagine_bench.envs.metaworld.meta_utils import MetaWrapper, baseline_env_name_list, rephrase_level_env_name_list, easy_level_env_name_list, hard_level_env_name_list
from imagine_bench.envs.metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from imagine_bench.envs.metaworld.meta_utils import TAU_LEN, get_noisy_entity_list, obs_online2noisy_offline, num_noisy_entity

import gymnasium as gym

gd_max_tau_len = 50
gd_batch_size = 100
ds_type_list = [
    'baseline',
    'rephrase_level',
    'easy_level',
    'hard_level',
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_type", type=str, default="rephrase", choices=['train', 'rephrase', 'easy', 'hard'], help="The type of offlineRL dataset.")
    parser.add_argument("--algo", type=str, default="cql", choices=['bc', 'cql', 'bcq', 'sac'], help="The name of offlineRL agent.")
    parser.add_argument("--dataset_path", type=str, default="./data/meta_world.hdf5", help="The path of offlineRL dataset.")
    parser.add_argument("--imagine_dataset_path", type=str, default='none', help="The path of imagine offlineRL dataset.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device of offlineRL training.")
    # offlineRL algorithm hyperparameters
    parser.add_argument("--seed", type=int, default=7, help="Seed.")
    # CQL
    parser.add_argument("--cql_alpha", type=float, default=10.0, help="Weight of conservative loss in CQL.")

    parser.add_argument("--agent_name")
    parser.add_argument("--env", type=str, default="Mujoco-v0", help="The name of Env.")

    args = parser.parse_args()

    args.agent_name = args.algo

    return args


def eval_given_level_agent(
    args1: argparse.Namespace,
    env: gym.Env,
    env_name: str,
    model_path: str,
    inst_list: list,
    agent: QLearningAlgoBase,
    epoch: int
):
    policy = agent

    eval_result_list = []
    success_mean = 0.0

    env_prefix = env_name[:env_name.find('-v2')]
    eval_sample_num = 11
    np.random.seed(args1.seed)
    range_tqdm = tqdm(range(eval_sample_num), ncols=120, leave=False)
    range_tqdm.set_description(f"Evaluating {args1.level}-{env_prefix} using {args1.agent_name.upper()}")
    succ_list = []
    reward_list = []
    for i in range_tqdm:
        online_obs = env.reset()[0]
        inst_encoding_idx = np.random.randint(low=0, high=len(inst_list))  # sample a inst uniformly
        inst_encoding = inst_list[inst_encoding_idx].flatten()
        eval_result = {
            'done': np.array([False]),
            'success': np.array([False]),
            'failure': np.array([False]),
        }
        noisy_entity_list = get_noisy_entity_list(env_name=env_name)
        tau_noisy_entity_list = np.random.choice(noisy_entity_list, size=num_noisy_entity)
        traj_reward = 0
        for step in range(env.max_path_length):

            # offline_obs = obs_online2offline(env_name, online_obs)
            offline_obs = obs_online2noisy_offline(env_name, online_obs, tau_noisy_entity_list=tau_noisy_entity_list)

            env_obs = offline_obs.copy()
            policy_obs = np.r_[env_obs, inst_encoding]
            action = policy.predict(policy_obs.reshape(1, -1)).flatten()
            next_obs, reward, terminated, truncated, info = env.step(action)
            traj_reward += reward
            terminated = terminated or bool(info['success'])
            done = terminated or truncated

            eval_result['done'] = np.array([done])
            eval_result['success'] = np.array([info['is_success']])
            eval_result['failure'] = np.array([done and not info['is_success']])

            online_obs = next_obs

            if done:
                succ_list.append(1 if bool(info['success']) else 0)
                reward_list.append(traj_reward)
                break

        eval_result_list.append(eval_result)
        success_mean = (success_mean * i + eval_result['success'].item()) / (i + 1)
        range_tqdm.set_postfix_str(f"Avg SR: {success_mean:.5f}")
    
    return succ_list, reward_list


def EvalCallBack(agent: QLearningAlgoBase, epoch: int, total_step: int) -> None:
    args1 = deepcopy(args)

    inst_encoding_dict = np.load(os.path.join(DATASET_PATH, 'meta_instructions_encoding.npy'), allow_pickle=True).item()

    wrap_info = {
        'reward_shaping': True
    }

    if args.ds_type != 'train':
        test_level_list = ['baseline', f'{args1.ds_type}_level']
    else:
        test_level_list = ['baseline', 'rephrase_level', 'easy_level', 'hard_level']
    
    for test_level in test_level_list:
        args1.level = test_level

        env_name_list = []
        if args1.level == 'baseline':
            env_name_list = baseline_env_name_list.copy()
        elif args1.level == 'rephrase_level':
            env_name_list = rephrase_level_env_name_list.copy()
        elif args1.level == 'easy_level':
            env_name_list = easy_level_env_name_list.copy()
        elif args1.level == 'hard_level':
            env_name_list = hard_level_env_name_list.copy()
        else:
            raise NotImplementedError
        
        succ_list = []
        reward_list = []
        for env_name in env_name_list:
            eval_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
            eval_env._freeze_rand_vec = False  # random reset
            eval_env.max_path_length = TAU_LEN  # Try to avoid returns of successful tau less than failed tau
            eval_env = MetaWrapper(eval_env, wrap_info=wrap_info)
            inst_list = inst_encoding_dict[env_name]
            result = eval_given_level_agent(
                args1=args1,
                env=eval_env,
                env_name=env_name,
                model_path=None,
                inst_list=inst_list,
                agent = agent,
                epoch = epoch
            )
            succ_list.extend(result[0])
            reward_list.extend(result[1])
        
        agent.logger.add_metric(f'eval/{args1.level}_succ', np.mean(succ_list))
        agent.logger.add_metric(f'eval/{args1.level}_reward', np.mean(reward_list))


if __name__ == '__main__':
    args = get_args()
    kwargs = vars(args)

    env_name = args.env
    level = args.ds_type
    if level == "train":
        env = imagine_bench.make(env_name, level='real')
    else:
        env = imagine_bench.make(env_name, level=level)
    
    if level == "train":
        real_data, _ = env.get_dataset(level="rephrase") 
        dataset = make_d3rlpy_dataset(real_data, None)
    else:
        real_data, imaginary_rollout = env.get_dataset(level=level) 
        dataset = make_d3rlpy_dataset(real_data, imaginary_rollout)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    register_encoder_factory(LlataEncoderFactory)
    if args.agent_name == 'bc':
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.BCConfig(
            encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'cql':
        alg_hyper_list = [
            'cql_alpha',
        ]
        agent = d3rlpy.algos.CQLConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            conservative_weight=args.cql_alpha,
        ).create(device=args.device)
    elif args.agent_name == 'awac':
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.AWACConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'td3+bc':
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.TD3PlusBCConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'bcq':
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.BCQConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    else:
        raise NotImplementedError

    exp_name = 'meta'
    kwargs = vars(args)    
    exp_name = f'{exp_name}_{"i-" if level != "train" else ""}{kwargs["ds_type"]}'
    exp_name_temp = f'{exp_name}_{kwargs["agent_name"]}_seed{kwargs["seed"]}'
    exp_name = f'{exp_name_temp}_{datetime.now().strftime("%m-%d_%H-%M-%S")}'

    # offline training
    agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name=exp_name,
        with_timestamp=False,
        logger_adapter=TensorboardAdapterFactory(root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        # callback=EvalCallBack,
        epoch_callback=EvalCallBack,
    )
