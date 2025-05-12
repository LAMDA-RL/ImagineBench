from typing import List, Dict

import numpy as np
from minigrid.wrappers import ImgObsWrapper

from envs import RIMAROEnv, LEVEL_LIST, download_dataset_from_url
from envs.babyai.env import MyGrid, EncodeWrapper, MyWrapper
from envs.babyai.grid_utils import baseline_env_name_list, rephrase_level_env_name_list, easy_level_env_name_list, hard_level_env_name_list
from envs.babyai.env import (
    GoToEnv, OpenEnv, PickUpEnv, PutNextEnv,
    GotoSeqEnv, PickUpSeqEnv, GoSEnv, GoTEnv,
    OpenLockEnv, PutLineEnv, PutPileEnv
)


class BabyAIEnv(RIMAROEnv):
    def __init__(self, **kwargs):
        self.dataset_url_dict = kwargs['dataset_url_dict']

        self.level = kwargs['level']
        if self.level == 'real':
            self.env_name_list = baseline_env_name_list.copy()
        elif self.level == 'rephrase_level':
            self.env_name_list = rephrase_level_env_name_list.copy()
        elif self.level == 'easy_level':
            self.env_name_list = easy_level_env_name_list.copy()
        elif self.level == 'hard_level':
            self.env_name_list = hard_level_env_name_list.copy()
        else:
            raise NotImplementedError
        
        self.env_list: List[MyGrid] = []
        for env_name in self.env_name_list:
            if self.level in ['real', 'rephrase_level']:
                if env_name == 'goto':
                    env = GoToEnv()
                elif env_name == 'open':
                    env = OpenEnv()
                elif env_name == 'pickup':
                    env = PickUpEnv()
                elif env_name == 'putnext':
                    env = PutNextEnv()
                else:
                    raise NotImplementedError
            elif self.level == 'easy':
                if env_name == 'goto_seq':
                    env = GotoSeqEnv()
                elif env_name == 'pickup_seq':
                    env = PickUpSeqEnv()
                elif env_name == 'go_straight':
                    env = GoSEnv()
                elif env_name == 'go_turn':
                    env = GoTEnv()
                else:
                    raise NotImplementedError
            elif self.level == 'hard':
                if env_name == 'open_lock':
                    env = OpenLockEnv()
                elif env_name == 'put_line':
                    env = PutLineEnv()
                elif env_name == 'put_pile':
                    env = PutPileEnv()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            env = ImgObsWrapper(env)
            env = EncodeWrapper(env)
            env = MyWrapper(env)
            self.env_list.append(env)
        
        self.action_space = self.env_list[0].action_space
        
        self.ptr = None
        self.path_dict = {}

        # lky: for debug
        self.path_dict['real'] = '/mnt/data/lky/data/BabyAI/babyai1.npy'
        self.path_dict['rephrase'] = '/home/liky/project/KALM_BabyAI/imagine_data_processed/imagine_processed_rephrase_04-06_15-27-37.npy'
        self.path_dict['easy'] = '/home/liky/project/KALM_BabyAI/imagine_data_processed/imagine_processed_easy_04-06_17-28-32.npy'
        self.path_dict['hard'] = '/home/liky/project/KALM_BabyAI/imagine_data_processed/imagine_processed_hard_04-06_19-56-03.npy'

    def reset(self, **kwargs):
        if self.ptr is None:
            self.ptr = 0
        else:
            self.ptr = (self.ptr + 1) % len(self.env_list)
        curr_env = self.env_list[self.ptr]

        return curr_env.reset(**kwargs)
    
    def step(self, action: int):
        curr_env: MyGrid = self.env_list[self.ptr]
        return curr_env.step(action)
    
    def get_dataset(self, level: str = 'rephrase') -> Dict[str, np.ndarray]:
        self.level = level

        assert self.level in LEVEL_LIST, f'level should be in {LEVEL_LIST}, but got {self.level}'

        if 'real' not in self.path_dict.keys():
            self.path_dict['real'] = download_dataset_from_url(self.dataset_url_dict['real'])
        real_dataset_path = self.path_dict['real']
        np_data = np.load(real_dataset_path, allow_pickle=True).item()
        real_dataset = {
                'masks': np_data['masks'][:],
                'observations': np_data['observations'][:],
                'actions': np_data['actions'][:],
                'rewards': np_data['rewards'][:],
            }
        
        if self.level not in self.path_dict.keys():
            self.path_dict[self.level] = download_dataset_from_url(self.dataset_url_dict[self.level])
        imaginary_level_dataset_path = self.path_dict[self.level]
        np_data = np.load(imaginary_level_dataset_path, allow_pickle=True).item()
        imaginary_level_dataset = {
            'masks': np_data['masks'][:],
            'observations': np_data['observations'][:],
            'actions': np_data['actions'][:],
            'rewards': np_data['rewards'][:],
        }

        return real_dataset, imaginary_level_dataset
