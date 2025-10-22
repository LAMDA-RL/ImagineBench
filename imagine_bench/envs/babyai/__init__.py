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

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['is_success'] = terminated
        return obs, reward, terminated, truncated, info
    
    def get_dataset(self, level: str = 'rephrase', data_model: str = 'llama2') -> Tuple[Dict[str, np.ndarray], Union[Dict[str, np.ndarray], None]]:
        real_dataset, imaginary_level_dataset = super().get_dataset(level=level, data_model=data_model)

        encoding = np.array([self.env.inst2encode[inst[0]] for inst in real_dataset['instructions']])
        observations: np.ndarray = real_dataset['observations']
        encoding = encoding[:, np.newaxis, :].repeat(observations.shape[1], axis=1)
        observations = np.concatenate([observations, encoding], axis=-1)
        real_dataset['observations'] = observations
        
        if level != 'real':
            encoding = np.array([self.env.inst2encode[inst[0]] for inst in imaginary_level_dataset['instructions']])
            observations: np.ndarray = imaginary_level_dataset['observations']
            encoding = encoding[:, np.newaxis, :].repeat(observations.shape[1], axis=1)
            observations = np.concatenate([observations, encoding], axis=-1)
            imaginary_level_dataset['observations'] = observations

        return real_dataset, imaginary_level_dataset
