import os
import sys
import random
import urllib.request
from typing import List
from operator import itemgetter
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import h5py
import gym.spaces
import numpy as np
import gym.spaces.box

from envs import DATASET_PATH, LEVEL_LIST, RIMAROEnv, download_dataset_from_url, show_progress
from clevr_robot_env import LlataEnv
from clevr_utils import terminal_fn_with_level, CLEVR_QPOS_OBS_INDICES, compute_tau_reward_arr, compute_task_reward_arr


level2true_level = {
    'real': 'tau_level',
    'rephrase': 'tau_level',
    'easy': 'step_level_a',
    'hard': 'task_level',
}


class BallEnv(RIMAROEnv, LlataEnv):
    def __init__(self, **kwargs):
        self.dataset_url_dict = kwargs['dataset_url_dict']

        self.max_episode_steps = 50
        self.num_object = 5
        self.level = kwargs['level']
        self.hist_obs_list = None
        self.timestep = 0
        self.ptr = None

        # 指令以及其余信息统一存为 npy, 数据集(obs, action, mask, reward)统一存为 h5
        self.path_dict = {}
        # self.init_dataset()
        
        # self.path_dict['real_npy'] = download_dataset_from_url(self.dataset_url_dict['real_npy'])
        # real_data = np.load(self.path_dict['real_npy'], allow_pickle=True).item()
        # self.real_data_info = {
        #     'instructions': real_data['instructions'],
        #     'goals': real_data['goals'],
        # }
        # self.imaginary_data_info = {}
        # imaginary_rephrase_data = np.load(self.path_dict['imaginary_rephrase_npy'], allow_pickle=True).item()
        # self.imaginary_data_info['rephrase_level'] = {
        #     'instructions': imaginary_rephrase_data['instructions'],
        #     'goals': imaginary_rephrase_data['goals'],
        # }
        # imaginary_easy_data = np.load(self.path_dict['imaginary_easy_npy'], allow_pickle=True).item()
        # self.imaginary_data_info['easy_level'] = {
        #     'instructions': imaginary_easy_data['instructions'],
        #     'goals': imaginary_easy_data['goals'],
        # }
        # imaginary_hard_data = np.load(self.path_dict['imaginary_hard_npy'], allow_pickle=True).item()
        # self.imaginary_data_info['hard_level'] = {
        #     'instructions': imaginary_hard_data['instructions'],
        #     'goals': imaginary_hard_data['goals'],
        # }

        self.prepare_test()

        data = np.load(DATASET_PATH.joinpath(f'clevr_test_{self.level+"_level" if self.level != "real" else "baseline"}.npy'), allow_pickle=True).item()
        self.observations: List[np.ndarray] = itemgetter("observations")(data)
        self.instructions = itemgetter("instructions")(data)
        self.goals: List[np.ndarray] = data["goals"] if "goals" in data else None

        self.inst_encoding = self.observations[0][0][2 * self.num_object:]

        LlataEnv.__init__(
            self,
            maximum_episode_steps=self.max_episode_steps,
            action_type='perfect',
            obs_type='order_invariant',
            use_subset_instruction=True,
            num_object=self.num_object,
            direct_obs=True,
            use_camera=False,
        )

        new_image_space = gym.spaces.Box(
            low=gym.spaces.box.get_inf(self.observations[0].dtype, sign='-'),
            high=gym.spaces.box.get_inf(self.observations[0].dtype, sign='+'),
            shape=(778, ),
            dtype=self.observations[0].dtype,
        )

        self.observation_space = new_image_space
        
    def reset(self, **kwargs):
        self.ptr = random.randint(0, len(self.observations) - 1)
        obss = self.observations[self.ptr]
        
        obs = LlataEnv.reset(self, **kwargs)
        init_env_obs = obss[0][:2 * self.num_object]
        self.inst_encoding = obss[0][2 * self.num_object:]
        qpos, qvel = self.physics.data.qpos.copy(), self.physics.data.qvel.copy()
        qpos[CLEVR_QPOS_OBS_INDICES(self.num_object)] = init_env_obs
        self.set_state(qpos, qvel)

        self.hist_obs_list = [init_env_obs]

        obs = self.get_obs()
        env_obs = obs[:2 * self.num_object]
        policy_obs = np.r_[env_obs, self.inst_encoding]

        self.timestep = 0
        self.prev_completed_goal_cnt = 0

        if len(self.goals[self.ptr].shape) == 2:
            self.single_goal = self.goals[self.ptr][0]
        elif len(self.goals[self.ptr].shape) == 1:
            self.single_goal = self.goals[self.ptr]
        else:
            raise NotImplementedError

        return policy_obs, {}

    def step(self, action):
        pre_env_obs = self.get_obs()[:2 * self.num_object]
        next_obs, reward, done, info = LlataEnv.step(self, action)
        env_obs = self.get_obs()[:2 * self.num_object]

        true_level = level2true_level[self.level]
        if self.hist_obs_list is None:
            eval_result = {
                'done': False,
                'success': np.array([False]),
            }
        else:
            terminal_kwargs = dict(
                insts=None,
                observations=np.array([next_obs]),
                number_of_objects=self.num_object,
                goals=np.array([self.single_goal]),
                level=true_level,
            )
            self.hist_obs_list.append(env_obs)
            if 'step_level' in true_level:
                terminal_kwargs['hist_observations'] = np.array(self.hist_obs_list).reshape(1, 2, 2 * self.num_object)
                terminal_kwargs['actions'] = np.array([action]).reshape(1, -1)
            eval_result = terminal_fn_with_level(**terminal_kwargs)

        self.timestep += 1

        terminated = eval_result['done']
        trunated = self.timestep >= self.max_episode_steps
        if info is not None:
            info['is_success'] = bool(eval_result['success'].item())
        else:
            info = {'is_success': bool(eval_result['success'].item())}

        if self.hist_obs_list is None:
            reward = 0.0
        else:
            if self.level in ['real', 'rephrase']:
                reward = compute_tau_reward_arr(tau_obs_arr=pre_env_obs[np.newaxis, ...], tau_next_obs_arr=env_obs[np.newaxis, ...], goal_arr=self.single_goal, is_success=False)
                reward = float(reward[0][0])
            elif self.level == 'easy':
                reward = 1.0 if info['is_success'] else 0.0
            elif self.level == 'hard':
                reward, self.prev_completed_goal_cnt = compute_task_reward_arr(
                    tau_obs_arr=pre_env_obs[np.newaxis, ...],
                    tau_next_obs_arr=env_obs[np.newaxis, ...],
                    goal_arr=self.single_goal,
                    is_success=False,
                    prev_completed_goal_cnt=self.prev_completed_goal_cnt
                    )
                reward = float(reward[0][0])
            else:
                raise NotImplementedError
        
        policy_obs = np.r_[env_obs, self.inst_encoding]

        return policy_obs, reward, terminated, trunated, info
    
    def init_dataset(self):
        if self.dataset_url_dict is None:
            raise ValueError("Offline env not configured with a dataset URL.")

        self.path_dict = {}
        for key, url in self.dataset_url_dict.items():
            self.path_dict[key] = download_dataset_from_url(url)

    def get_dataset(self, **kwargs):
        self.level = kwargs.get('level', 'rephrase')

        assert self.level in LEVEL_LIST, f'level should be in {LEVEL_LIST}, but got {self.level}'

        if 'real_h5' not in self.path_dict.keys():
            self.path_dict['real_h5'] = download_dataset_from_url(self.dataset_url_dict['real_h5'])
        real_dataset_path = self.path_dict['real_h5']
        with h5py.File(real_dataset_path, 'r') as f:
            real_dataset = {
                'masks': f['masks'][:],
                'observations': f['observations'][:],
                'actions': f['actions'][:],
                'rewards': f['rewards'][:],
            }
        
        if f'imaginary_{self.level}_h5' not in self.path_dict.keys():
            self.path_dict[f'imaginary_{self.level}_h5'] = download_dataset_from_url(self.dataset_url_dict[f'imaginary_{self.level}_h5'])
        imaginary_level_dataset_path = self.path_dict[f'imaginary_{self.level}_h5']
        with h5py.File(imaginary_level_dataset_path, 'r') as f:
            imaginary_level_dataset = {
                'masks': f['masks'][:],
                'observations': f['observations'][:],
                'actions': f['actions'][:],
                'rewards': f['rewards'][:],
            }
        
        return real_dataset, imaginary_level_dataset
    
    def get_instruction(self):
        # 存在多条语义相同指令
        instruction = np.random.choice(self.data_info['instructions'][self.ptr])

        return instruction
    
    def prepare_test(self):
        test_data_url_dict = {
            'baseline': 'https://box.nju.edu.cn/f/4618c2d3ec614d07937f/?dl=1',
            'rephrase_level': 'https://box.nju.edu.cn/f/ac679a8e7e9745d5b392/?dl=1',
            "easy_level": 'https://box.nju.edu.cn/f/b0731b98de1a46d0bbfa/?dl=1',
            "hard_level": 'https://box.nju.edu.cn/f/7280d5c025e4419a86b0/?dl=1'
        }
        for level in ['baseline', 'rephrase_level', 'easy_level', 'hard_level']:
            test_data_path = os.path.join(DATASET_PATH, f'clevr_test_{level}.npy')
            dataset_url = test_data_url_dict[level]
            if not os.path.exists(test_data_path):
                urllib.request.urlretrieve(dataset_url, test_data_path, show_progress)
                if not os.path.exists(test_data_path):
                    raise IOError("Failed to download dataset from %s" % dataset_url)
