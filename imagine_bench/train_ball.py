import os
os.environ["MUJOCO_GL"]='egl'

import random
import argparse
from typing import List, Optional
from copy import deepcopy
from datetime import datetime
from operator import itemgetter

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

import imagine_bench
from algo import d3rlpy
from imagine_bench.envs import DATASET_PATH
from algo.d3rlpy.logging import TensorboardAdapterFactory
from algo.d3rlpy.algos.qlearning import QLearningAlgoBase
from imagine_bench.envs.ball.clevr_robot_env import LlataEnv
from imagine_bench.utils import LlataEncoderFactory, make_d3rlpy_dataset
from imagine_bench.envs.ball.utils.clevr_utils import terminal_fn_with_level, CLEVR_QPOS_OBS_INDICES


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_type", type=str, default="rephrase", choices=['train', 'rephrase', 'easy', 'hard'], help="The type of offlineRL dataset.")
    parser.add_argument("--algo", type=str, default="cql", choices=['bc', 'cql', 'bcq', 'sac'], help="The name of offlineRL agent.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device for offlineRL training.")
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
    env: LlataEnv,
    num_obj: int,
    max_traj_len: int,
    model_path: str,
    instructions: List[str],
    observations: List[np.ndarray],
    terminals: Optional[List[np.ndarray]] = None,
    goals: Optional[List[np.ndarray]] = None,
    actions: Optional[List[np.ndarray]] = None,
    success: Optional[List[np.ndarray]] = None,
    agent: QLearningAlgoBase = None,
    epoch: int = None,
):
    assert len(observations) == len(instructions), "instructions, observations number mismatch"
    if goals is not None:
        assert len(observations) == len(goals), \
            "instructions, observations, goals number mismatch"
    if terminals is not None:
        assert len(observations) == len(terminals), \
            "observations, terminals number mismatch"
    if actions is not None:
        # for recording only
        assert len(observations) == len(actions) or len(observations) == len(actions) + 1, \
            "observations, actions number mismatch"
    if success is not None:
        # for recording only
        assert len(observations) == len(success), \
            "observations, success number mismatch"
        
    policy = agent
    args1.level = transfer2old[args1.level]
    if 'step_level' in args1.level:
        eval_sample_num = 666
    elif args1.level in ['baseline', 'tau_level', 'task_level']:
        eval_sample_num = 111
    else:
        raise NotImplementedError

    assert eval_sample_num <= len(instructions), "eval_sample_num should be less than the number of instructions"
    np.random.seed(args1.seed)
    sampled_indices = np.random.choice(len(instructions), eval_sample_num, replace=False)
    range_tqdm = tqdm(sampled_indices, ncols=120, leave=False)
    range_tqdm.set_description(f"Epoch:{epoch}, Train:{args.ds_type}, Eval:{args1.level}")
    succ_list = []
    reward_list = []
    succ_len_list = []
    succ_inst_list = []
    fail_inst_list = []
    succ_goal_list = []
    fail_goal_list = []
    for i in range_tqdm:
        insts = instructions[i]
        obss = observations[i]
        if len(goals[i].shape) == 2:
            single_goal = goals[i][0]
        elif len(goals[i].shape) == 1:
            single_goal = goals[i]
        else:
            raise NotImplementedError
        try:
            single_inst = np.random.choice(insts)
        except:
            single_inst = insts
        
        # Reset Env to dataset obs
        env.reset()
        init_env_obs = obss[0][:2 * num_obj]
        inst_encoding = obss[0][2 * num_obj:]
        qpos, qvel = env.physics.data.qpos.copy(), env.physics.data.qvel.copy()
        qpos[CLEVR_QPOS_OBS_INDICES(num_obj)] = init_env_obs
        env.set_state(qpos, qvel)

        hist_obs_list = [init_env_obs]
        eval_result = {
            'done': np.array([False]),
            'success': np.array([False]),
            'failure': np.array([False]),
        }
        traj_reward = 0
        for step in range(max_traj_len):
            obs = env.get_obs()
            env_obs = obs[:2 * num_obj]
            policy_obs = np.r_[env_obs, inst_encoding]
            action = policy.predict(policy_obs.reshape(1, -1)).flatten()
            next_obs, reward, done, info = env.step(action)
            traj_reward += reward
            next_env_obs = next_obs[:2 * num_obj]
            hist_obs_list.append(next_env_obs)

            if test_render:
                rgb_array = env.render(mode='rgb_array')
                im = Image.fromarray(rgb_array)
                temp_dir = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp/fig_demo'), f'{time_str}')
                os.makedirs(name=temp_dir, exist_ok=True)
                im.save(os.path.join(temp_dir, f'{args1.level}_{i}_{step}.png'))
            
            terminal_kwargs = dict(
                insts=[single_inst],
                observations=np.array([next_env_obs]),
                number_of_objects=num_obj,
                goals=np.array([single_goal]),
                level=args1.level,
            )
            if 'step_level' in args1.level:
                terminal_kwargs['hist_observations'] = np.array(hist_obs_list).reshape(1, 2, 2 * num_obj)
                terminal_kwargs['actions'] = np.array([action]).reshape(1, -1)
                terminal_kwargs['level'] = 'step_level_a'
            if args1.level == 'baseline':
                terminal_kwargs['level'] = 'tau_level'

            eval_result = terminal_fn_with_level(**terminal_kwargs)
            done = eval_result['done']
            if done.item() or (step==max_traj_len-1):
                succ_list.append(1 if eval_result['success'].item() else 0)
                reward_list.append(traj_reward)
                # single_goal_temp = list(single_goal)
                # single_goal_temp.sort()
                single_goal_temp = [single_goal[num_obj]]
                if eval_result['success'].item():
                    succ_len_list.append(step+1)
                    if ('hard' in args1.level) or ('task' in args1.level):
                        succ_inst_list.append(insts[0])
                        succ_goal_list.append(tuple(single_goal_temp))
                else:
                    if ('hard' in args1.level) or ('task' in args1.level):
                        fail_inst_list.append(insts[0])
                        fail_goal_list.append(tuple(single_goal_temp))
                range_tqdm.set_postfix({'succ': np.mean(succ_list), 'len_s': np.mean(succ_len_list) if len(succ_len_list) > 0 else 0})
                break
    
    if ('hard' in args1.level) or ('task' in args1.level):
        from collections import Counter
        succ_counter = Counter(succ_goal_list)
        fail_counter = Counter(fail_goal_list)
        pass

    agent.logger.add_metric(f'eval/{args1.level}_succ', np.mean(succ_list))
    # agent.logger.add_metric(f'eval/{args1.level}_reward', np.mean(reward_list))
    agent.logger.add_metric(f'eval/{args1.level}_len_s', np.mean(succ_len_list) if len(succ_len_list) > 0 else 0)


def EvalCallBack(agent: QLearningAlgoBase, epoch: int, total_step: int) -> None:   
    args1 = deepcopy(args)
    args1.max_traj_len = 50
    args1.render = test_render
    max_traj_len = args1.max_traj_len

    if args.ds_type != 'train':
        test_level_list = ['baseline', f'{args1.ds_type}_level']
    else:
        test_level_list = ['baseline', 'rephrase_level', 'easy_level', 'hard_level']
        
    for test_level in test_level_list:
        args1.level = test_level
        data_path = os.path.join(DATASET_PATH, f'clevr_test_{args1.level}.npy')
        data = np.load(data_path, allow_pickle=True).item()

        instructions, observations, num_obj = itemgetter(
            "instructions", "observations", "number_of_objects")(data)
        goals = data["goals"] if "goals" in data else None
        terminals = data["terminals"] if "terminals" in data else None
        actions = data["actions"] if "actions" in data else None
        success = data["success"] if "success" in data else None
        env = LlataEnv(
            maximum_episode_steps=max_traj_len,
            action_type='perfect',
            obs_type='order_invariant',
            use_subset_instruction=True,
            num_object=num_obj,
            direct_obs=True,
            use_camera=args1.render,
        )
        eval_given_level_agent(
            args1=args1,
            env=env,
            num_obj=num_obj,
            max_traj_len=max_traj_len,
            model_path=None,
            instructions=instructions,
            observations=observations,
            terminals=terminals,
            goals=goals,
            actions=actions,
            success=success,
            agent = agent,
            epoch = epoch
        )


if __name__ == '__main__':
    args = get_args()
    transfer2old = {
        'baseline': 'baseline',
        'rephrase_level': 'tau_level',
        'easy_level': 'step_level',
        'hard_level': 'task_level',
    }
    kwargs = vars(args)

    env_name = args.env
    level = args.ds_type
    if level == "train":
        env = imagine_bench.make(env_name, level='real')
    else:
        env = imagine_bench.make(env_name, level=level)
    
    env.prepare_test()

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
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.DiscreteBCConfig(
            encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'cql':
        alg_hyper_list = [
            'cql_alpha',
        ]
        agent = d3rlpy.algos.DiscreteCQLConfig(
            encoder_factory=LlataEncoderFactory(feature_size=64, hidden_size=64),
            alpha=args.cql_alpha,
        ).create(device=args.device)
    elif args.agent_name == 'bcq':
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.DiscreteBCQConfig(
            encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'sac':
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.DiscreteSACConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    else:
        raise NotImplementedError

    exp_name = 'ball'
    kwargs = vars(args)    
    exp_name = f'{exp_name}_{"i-" if level != "train" else ""}{kwargs["ds_type"]}'
    exp_name_temp = f'{exp_name}_{kwargs["agent_name"]}_seed{kwargs["seed"]}'
    time_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    exp_name = f'{exp_name_temp}_{time_str}'

    test_render = False
    # test_render = True

    # offline training
    agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name=exp_name,
        with_timestamp=False,
        logger_adapter=TensorboardAdapterFactory(root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        epoch_callback=EvalCallBack,
    )
