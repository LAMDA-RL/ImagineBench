import os
import random
import argparse
from datetime import datetime
from typing import Dict, List

import torch
import gymnasium
import numpy as np
from tqdm import tqdm

import imagine_bench
from algo import d3rlpy
from algo.d3rlpy.logging import TensorboardAdapterFactory
from algo.d3rlpy.algos.qlearning import QLearningAlgoBase
from imagine_bench.utils import LlataEncoderFactory, make_d3rlpy_dataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_type", type=str, default="train", choices=['train', 'rephrase', 'easy', 'hard'], help="The type of offlineRL dataset.")
    parser.add_argument("--algo", type=str, default="cql", choices=['bc', 'cql', 'bcq', 'sac'], help="The name of offlineRL agent.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device for offlineRL training.")
    # offlineRL algorithm hyperparameters
    parser.add_argument("--seed", type=int, default=7, help="Seed.")
    # CQL
    parser.add_argument("--cql_alpha", type=float, default=10.0, help="Weight of conservative loss in CQL.")

    parser.add_argument("--eval_episodes", type=int, default=40)

    parser.add_argument("--agent_name")
    parser.add_argument("--env", type=str, default="Mujoco-v0", help="The name of Env.")

    args = parser.parse_args()

    args.agent_name = args.algo

    return args


def EvalCallBack(agent: QLearningAlgoBase, epoch: int, total_step: int) -> None:
    if args.ds_type != 'train':
        test_level_list = ['real', args.ds_type]
    else:
        test_level_list = ['real', 'rephrase', 'easy', 'hard']
    
    env_dict: Dict[str, gymnasium.Env] = {}
    for test_level in test_level_list:
        env_dict[test_level] = imagine_bench.make('BabyAI-v0', level=test_level)

    succ_dict: Dict[str, List[int]] = {}
    reward_dict: Dict[str, List[float]] = {}
    for test_level, env in env_dict.items():
        obs, _ = env.reset()
        obs: np.ndarray

        traj_reward = 0
        traj_len = 0
        
        succ_list = []
        reward_list = []
        traj_len_list = []

        pbar = tqdm(total=args.eval_episodes, desc=f'epoch {epoch} eval: {test_level}', ncols=120, leave=False)
        while True:
            action = agent.predict(obs[np.newaxis, ...])
            action = int(action[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            traj_reward += reward
            traj_len += 1

            if terminated or truncated:
                succ_list.append(1 if terminated else 0)
                reward_list.append(traj_reward)
                traj_len_list.append(traj_len)
                
                pbar.update()

                if len(succ_list) == args.eval_episodes:
                    break
                else:
                    obs, _ = env.reset()
                    traj_reward = 0
                    traj_len = 0
        pbar.close()
        
        agent.logger.add_metric(f'eval/{test_level}_succ', np.mean(succ_list))
        agent.logger.add_metric(f'eval/{test_level}_reward', np.mean(reward_list))
        agent.logger.add_metric(f'eval_len/{test_level}_len', np.mean(traj_len_list))

        for level_name, succ_list_local in succ_dict.items():
            agent.logger.add_metric(f'eval_{level_name}_succ', np.mean(succ_list_local))
            agent.logger.add_metric(f'eval_{level_name}_reward', np.mean(reward_dict[level_name]))


if __name__ == '__main__':
    args = get_args()

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

    if args.agent_name == 'bc':
        agent = d3rlpy.algos.DiscreteBCConfig(
            encoder_factory=LlataEncoderFactory(feature_size=64, hidden_size=64),
        ).create(device=args.device)
    elif args.agent_name == 'cql':
        agent = d3rlpy.algos.DiscreteCQLConfig(
            encoder_factory=LlataEncoderFactory(feature_size=64, hidden_size=64),
            alpha=args.cql_alpha,
        ).create(device=args.device)
    elif args.agent_name == 'bcq':
        agent = d3rlpy.algos.DiscreteBCQConfig(
            encoder_factory=LlataEncoderFactory(feature_size=64, hidden_size=64),
        ).create(device=args.device)
    elif args.agent_name == 'sac':
        agent = d3rlpy.algos.DiscreteSACConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=64, hidden_size=64),
            critic_encoder_factory=LlataEncoderFactory(feature_size=64, hidden_size=64),
        ).create(device=args.device)
    else:
        raise NotImplementedError

    exp_name = 'BabyAI'
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
        epoch_callback=EvalCallBack,
    )

    print("===> offline training finished")
