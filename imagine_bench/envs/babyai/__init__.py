import os
import sys
from typing import Dict, Tuple, Union

import numpy as np
from minigrid.wrappers import ImgObsWrapper

from envs import RIMAROEnv, LEVEL_LIST, download_dataset_from_url


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from base import MyWrapper, ActionWrapper, DenseRewardWrapper, EncodeWrapper, LanguageWrapper, MyGrid
from real import GoToEnv, OpenEnv, PickUpEnv, PutNextEnv
from easy import OpenGoEnv, OpenPickEnv, GoWallEnv, GoCenterEnv
from hard import OpenLockEnv, PutLineEnv, PutPileEnv


class BabyAIEnv(RIMAROEnv):
    def __init__(self, **kwargs):
        self.dataset_url_dict = kwargs['dataset_url_dict']

        self.level = kwargs['level']
        if self.level in ['real', 'rephrase']:
            env_list = [GoToEnv, OpenEnv, PickUpEnv, PutNextEnv]
        elif self.level == 'easy':
            env_list = [OpenGoEnv, OpenPickEnv, GoWallEnv, GoCenterEnv]
        elif self.level == 'hard':
            env_list = [OpenLockEnv, PutLineEnv, PutPileEnv]
        else:
            raise NotImplementedError
        
        for i in range(len(env_list)):
            env = env_list[i]()
            env = ActionWrapper(env)
            env = ImgObsWrapper(env)
            env = EncodeWrapper(env)
            env = DenseRewardWrapper(env)
            env = MyWrapper(env)
            env_list[i] = env
        self.env = LanguageWrapper(
            env_list=env_list,
            inst_encode_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'babyai_encode.npy'),
            level=self.level,
            use_gym=False
            )
        
        self.action_space = self.env.action_space
        
        self.path_dict = {}

        # lky: for debug
        self.path_dict['real'] = '/mnt/data/lky/data/RIMRO/babyai_real.npy'
        # self.path_dict['rephrase'] = '/mnt/data/lky/data/RIMRO/babyai_rephrase.npy'
        self.path_dict['easy'] = '/mnt/data/lky/data/RIMRO/babyai_easy.npy'
        self.path_dict['hard'] = '/mnt/data/lky/data/RIMRO/babyai_hard.npy'

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['is_success'] = terminated
        return obs, reward, terminated, truncated, info
    
    def get_dataset(self, level: str = 'rephrase') -> Tuple[Dict[str, np.ndarray], Union[Dict[str, np.ndarray], None]]:
        assert level in LEVEL_LIST, f'level should be in {LEVEL_LIST}, but got {self.level}'

        if 'real' not in self.path_dict.keys():
            self.path_dict['real'] = download_dataset_from_url(self.dataset_url_dict['real'])
        real_dataset_path = self.path_dict['real']
        np_data = np.load(real_dataset_path, allow_pickle=True).item()
        encoding = np.array([self.env.inst2encode[inst[0]] for inst in np_data['instructions']])
        observations: np.ndarray = np_data['observations']
        encoding = encoding[:, np.newaxis, :].repeat(observations.shape[1], axis=1)
        observations = np.concatenate([observations, encoding], axis=-1)
        real_dataset = {
                'masks': np_data['masks'][:],
                'observations': observations,
                'actions': np_data['actions'][:],
                'rewards': np_data['rewards'][:],
            }
        
        if level != 'real':
            if level not in self.path_dict.keys():
                self.path_dict[level] = download_dataset_from_url(self.dataset_url_dict[f'{level}'])
            imaginary_level_dataset_path = self.path_dict[level]
            np_data = np.load(imaginary_level_dataset_path, allow_pickle=True).item()
            encoding = np.array([self.env.inst2encode[inst[0]] for inst in np_data['instructions']])
            observations: np.ndarray = np_data['observations']
            encoding = encoding[:, np.newaxis, :].repeat(observations.shape[1], axis=1)
            observations = np.concatenate([observations, encoding], axis=-1)
            imaginary_level_dataset = {
                'masks': np_data['masks'][:],
                'observations': observations,
                'actions': np_data['actions'][:],
                'rewards': np_data['rewards'][:],
            }

            return real_dataset, imaginary_level_dataset
        else:
            return real_dataset, None
