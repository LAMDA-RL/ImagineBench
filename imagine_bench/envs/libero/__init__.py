import gym
from gym import spaces
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from Libero.libero.envs import OffScreenRenderEnv
from Libero.libero import benchmark
import random
import matplotlib.pyplot as plt
import cv2
import imageio
import torch
import torchvision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from envs.libero.env import level2true_level, baseline_env_name_list, rephrase_level_env_name_list, easy_level_env_name_list, hard_level_env_name_list, VectorLibero
from envs import RIMAROEnv, LEVEL_LIST, download_dataset_from_url

class LiberoEnv(RIMAROEnv):
    def __init__(self, **kwargs):
        self.dataset_url_dict = kwargs['dataset_url_dict']

        self.level = kwargs['level']
        true_level = level2true_level[self.level]
        if true_level == 'baseline':
            self.env_name_list = baseline_env_name_list.copy()
        elif true_level == 'rephrase_level':
            self.env_name_list = rephrase_level_env_name_list.copy()
        elif true_level == 'easy_level':
            self.env_name_list = easy_level_env_name_list.copy()
        elif true_level == 'hard_level':
            self.env_name_list = hard_level_env_name_list.copy()
        else:
            raise NotImplementedError
        # 一个level对应多个env
        self.env_list = []
        for env_name in self.env_name_list:
            eval_env = VectorLibero(env_name)
            self.env_list.append(eval_env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44 + 768,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        self.ptr = None
        self.path_dict = {}
        # self.path_dict['real'] = 'rimro/envs/data/libero_real.npy'
        # self.path_dict['rephrase'] = 'rimro/envs/data/libero_imaginary_rephrase.npy'
        # self.path_dict['easy'] = 'rimro/envs/data/libero_imaginary_easy.npy'
        # self.path_dict['hard'] = 'rimro/envs/data/libero_imaginary_hard.npy'
        self.inst2encode = np.load(os.path.dirname(__file__) + '/libero_files/libero_encode.npy', allow_pickle=True).item()
    def reset(self, **kwargs):
        if self.ptr is None:
            self.ptr = 0
        else:
            self.ptr = (self.ptr + 1) % len(self.env_list)
        curr_env = self.env_list[self.ptr]
        obs, _ = curr_env.reset(**kwargs)
        inst = random.choice(curr_env.get_instructions(curr_env.env_name))
        self.inst_encode = self.inst2encode[inst]
        return np.concatenate((obs, self.inst_encode), axis=0), {}
    
    def step(self, action):
        curr_env = self.env_list[self.ptr]
        obs, reward, terminated, truncated, info = curr_env.step(action)
        obs = np.concatenate([obs, self.inst_encode], axis=0)
        return obs, reward, terminated, truncated, info

    def get_dataset(self, level='rephrase'):
        self.level = level

        if 'real' not in self.path_dict.keys():
            self.path_dict['real'] = download_dataset_from_url(self.dataset_url_dict['real'])
        real_dataset_path = self.path_dict['real']
        np_data = np.load(real_dataset_path, allow_pickle=True).item()
        masks = np_data['masks'][:]
        observations = np_data['observations'][:]
        actions = np_data['actions'][:]
        rewards = np_data['rewards'][:]
        instructions = np_data['instructions'][:]
        encoding = np.array([self.inst2encode[inst[0]] for inst in instructions])
        encoding = encoding[:, np.newaxis, :].repeat(observations.shape[1], axis=1)
        observations = np.concatenate([observations, encoding], axis=-1)
        real_dataset = {
                'masks': masks,
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
            }
        
        if self.level not in self.path_dict.keys():
            self.path_dict[self.level] = download_dataset_from_url(self.dataset_url_dict[self.level])
        imaginary_level_dataset_path = self.path_dict[self.level]
        np_data = np.load(imaginary_level_dataset_path, allow_pickle=True).item()
        masks = np_data['masks'][:]
        observations = np_data['observations'][:]
        actions = np_data['actions'][:]
        rewards = np_data['rewards'][:]
        instructions = np_data['instructions'][:]
        encoding = np.array([self.inst2encode[inst[0]] for inst in instructions])
        encoding = encoding[:, np.newaxis, :].repeat(observations.shape[1], axis=1)
        observations = np.concatenate([observations, encoding], axis=-1)
        imaginary_level_dataset = {
            'masks': masks,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
        }
        return real_dataset, imaginary_level_dataset



if __name__ == "__main__":
    env = LiberoEnv(
        dataset_url_dict="none",
        level="rephrase",
    )
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(obs)
        print(next_obs)
        print(reward)
        print(done)
        print(info)
    real_dataset, easy_dataset = env.get_dataset(level='easy')
    # print(real_dataset)
    # print(easy_dataset)

    print("done")


