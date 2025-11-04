import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import h5py
import progressbar
import numpy as np
import pandas as pd
from tqdm import tqdm
from gymnasium import spaces
from huggingface_hub import hf_hub_download


DATASET_PATH = Path(__file__).parent.joinpath('data')
os.makedirs(DATASET_PATH, exist_ok=True)
ENV_ID_LIST = [
    'Ball-v0',
    'MetaWorld-v0',
    'BabyAI-v0',
    'Libero-v0',
    'Mujoco-v0'
]
LEVEL_LIST = [
    'real',
    'rephrase',
    'easy',
    'hard',
]


class RIMAROEnv:
    action_space: spaces.Box = None
    
    def __init__(self, **kwargs):
        self.path_dict = {}
        self.dataset_url_dict = {}

    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
        
    def init_dataset(self):
        raise NotImplementedError

    def get_dataset(self, level: str = 'rephrase', data_model: str = 'llama2') -> Tuple[Dict[str, np.ndarray], Union[Dict[str, np.ndarray], None]]:
        assert level in LEVEL_LIST, f'level should be in {LEVEL_LIST}, but got {self.level}'
        self.level = level

        if 'real' not in self.path_dict.keys():
            self.path_dict['real'] = download_dataset_from_url(self.dataset_url_dict['real'])
        real_dataset_path = self.path_dict['real']
        real_dataset_raw = load_parquet_dataset(real_dataset_path)
        real_dataset = {
                'masks': real_dataset_raw['masks'][:],
                'observations': real_dataset_raw['observations'][:],
                'actions': real_dataset_raw['actions'][:],
                'rewards': real_dataset_raw['rewards'][:],
            }
        if 'instructions' in real_dataset_raw:
            real_dataset['instructions'] = real_dataset_raw['instructions'][:]

        if level != 'real':
            if data_model == 'llama2':
                query_key = self.level
            elif data_model == 'qwen3':
                query_key = f'{self.level}_qwen'
            else:
                raise NotImplementedError(f"data_model {data_model} not supported.")
            if query_key not in self.path_dict.keys():
                self.path_dict[query_key] = download_dataset_from_url(self.dataset_url_dict[query_key])
            imaginary_level_dataset_path = self.path_dict[query_key]
            imaginary_level_dataset_raw = load_parquet_dataset(imaginary_level_dataset_path)
            imaginary_level_dataset = {
                'masks': imaginary_level_dataset_raw['masks'][:] if 'masks' in imaginary_level_dataset_raw else imaginary_level_dataset_raw['action_masks'][:],
                'observations': imaginary_level_dataset_raw['observations'][:],
                'actions': imaginary_level_dataset_raw['actions'][:],
                'rewards': imaginary_level_dataset_raw['rewards'][:],
            }
            if 'instructions' in imaginary_level_dataset_raw:
                imaginary_level_dataset['instructions'] = imaginary_level_dataset_raw['instructions'][:]
        else:
            imaginary_level_dataset = None
        
        return real_dataset, imaginary_level_dataset
    
    def get_instruction(self):
        raise NotImplementedError


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


# url2ds_name = {
#     # ball
#     'https://box.nju.edu.cn/f/c67fb5ed23694db0baaa/?dl=1': 'ball_imaginary_rephrase.h5',
#     'https://box.nju.edu.cn/f/e857ae10a53a4758a81f/?dl=1': 'ball_imaginary_easy.npy',
#     'https://box.nju.edu.cn/f/ac414bd4cf014dce87c6/?dl=1': 'ball_imaginary_hard.h5',
#     'https://box.nju.edu.cn/f/53e31be7ab9248e4b292/?dl=1': 'ball_imaginary_hard.npy',
#     'https://box.nju.edu.cn/f/1185b8b8673a47daae6c/?dl=1': 'ball_imaginary_easy.h5',
#     'https://box.nju.edu.cn/f/d3bceb9d5c5248d8b410/?dl=1': 'ball_imaginary_rephrase.npy',
#     'https://box.nju.edu.cn/f/e7465d994af04ffc9f21/?dl=1': 'ball_real.hdf5',
#     'https://box.nju.edu.cn/f/ede29bd4d9d74c93a6d4/?dl=1': 'ball_real.npy',
#     # metaworld
#     'https://box.nju.edu.cn/f/6b56624598f0487fb65a/?dl=1': 'metaworld_imaginary_easy.h5',
#     'https://box.nju.edu.cn/f/d2f9ec3ddf8c46c6a2e3/?dl=1': 'metaworld_imaginary_hard.h5',
#     'https://box.nju.edu.cn/f/5515ee501db948eb84db/?dl=1': 'metaworld_imaginary_rephrase.h5',
#     'https://box.nju.edu.cn/f/0fea8e468869468092e9/?dl=1': 'metaworld_real.h5',
#     # babyai
#     'https://box.nju.edu.cn/f/da1fc389e5d24c45a3a4/?dl=1': 'babyai_imaginary_easy.npy',
#     'https://box.nju.edu.cn/f/b8c6282e2ddf4819b972/?dl=1': 'babyai_imaginary_hard.npy',
#     'https://box.nju.edu.cn/f/e4d6695bcbe141bf927b/?dl=1': 'babyai_imaginary_rephrase.npy',
#     'https://box.nju.edu.cn/f/47ef43a660874409a420/?dl=1': 'babyai_real.npy',
#     # libero
#     'https://box.nju.edu.cn/f/1776405d68734731b96f/?dl=1': 'libero_imaginary_easy.npy',
#     'https://box.nju.edu.cn/f/8c1dce3a0ead47b6a67a/?dl=1': 'libero_imaginary_hard.npy',
#     'https://box.nju.edu.cn/f/f5dc1425fcb742428ef1/?dl=1': 'libero_imaginary_rephrase.npy',
#     'https://box.nju.edu.cn/f/93ef16b8e2d64f5ea935/?dl=1': 'libero_real.npy',
#     # mujoco
#     'https://box.nju.edu.cn/f/dbf3a096b380460bb1b9/?dl=1': 'mujoco_imaginary_easy.npy',
#     'https://box.nju.edu.cn/f/f70e67f27c8d40d59568/?dl=1': 'mujoco_imaginary_hard.npy',
#     'https://box.nju.edu.cn/f/c53bc2307c6d49ea9b0b/?dl=1': 'mujoco_imaginary_rephrase.npy',
#     'https://box.nju.edu.cn/f/44f5c558982e4dc3b5d5/?dl=1': 'mujoco_real.npy',
# }


BASE_URL = 'https://huggingface.co/datasets/NJU-RLer/ImagineBench/resolve/main'


def filepath_from_url(dataset_url: str):
    # dataset_name = url2ds_name[dataset_url][0]
    dataset_name = dataset_url.split('/')[-1].split('?')[0]
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download_dataset_from_url(dataset_url: str) -> str:
    dataset_filepath = filepath_from_url(dataset_url)

    # if not os.path.exists(dataset_filepath):
    #     print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
    #     urllib.request.urlretrieve(dataset_url, dataset_filepath, show_progress)
    # if not os.path.exists(dataset_filepath):
    #     raise IOError("Failed to download dataset from %s" % dataset_url)

    hf_hub_download(
        repo_id="NJU-RLer/ImagineBench",
        filename=dataset_url.split('/')[-1].split('?')[0],
        repo_type='dataset',
        local_dir=DATASET_PATH,
        )
    
    return dataset_filepath


# 实现类似 gym 的 make 函数
def make(env_id: str, **kwargs) -> RIMAROEnv:
    if env_id not in ENV_ID_LIST:
        raise ValueError(f"env_id {env_id} is not supported.")
    
    env = None
    kwargs['level'] = kwargs.get('level', 'real')
    env_name_lower = env_id.split('-')[0].lower()

    kwargs['dataset_url_dict'] = {}
    for level in LEVEL_LIST:
        kwargs['dataset_url_dict'][level] = f'{BASE_URL}/{env_name_lower}_{level}.parquet?download=true'
        if level != 'real':
            kwargs['dataset_url_dict'][f'{level}_qwen'] = f'{BASE_URL}/{env_name_lower}_{level}_qwen.parquet?download=true'

    if env_id == 'Ball-v0':
        from envs.ball import BallEnv
        env = BallEnv(**kwargs)
    elif env_id == 'MetaWorld-v0':
        from envs.metaworld import MetaWorldEnv
        env = MetaWorldEnv(**kwargs)
    elif env_id == 'BabyAI-v0':
        from envs.babyai import BabyAIEnv
        env = BabyAIEnv(**kwargs)
    elif env_id == 'Libero-v0':
        from envs.libero import LiberoEnv
        env = LiberoEnv(**kwargs)
    elif env_id == 'Mujoco-v0':
        from envs.mujoco import MujocoEnv
        env = MujocoEnv(**kwargs)
    else:
        raise NotImplementedError
    
    return env


def load_parquet_dataset(dataset_path: str) -> Dict[str, Union[np.ndarray, str, int, List[str], List[np.ndarray]]]:
    df_loaded = pd.read_parquet(dataset_path)
    loaded_data = {}

    array_keys = set()
    scalar_keys: Set[str] = set()
    list_array_keys = set()
    list_scalar_keys = set()
    
    for col in df_loaded.columns:
        if col.endswith('_data'):
            if not col.startswith('list_'):
                array_keys.add(col[:-5])  # remove "_data"
            else:
                list_array_keys.add(col[5:-5])  # remove "list_" and "_data"
        elif (not col.endswith(('_shape', '_dtype'))) and (col not in ['file_name', 'row_index']):
            if not col.startswith('list_'):
                scalar_keys.add(col)
            else:
                list_scalar_keys.add(col[5:])
    
    for key in scalar_keys:
        loaded_data[key] = df_loaded[key].iloc[0]
    
    for key in tqdm(array_keys, desc="loading arrays", leave=False):
        first_row = df_loaded.iloc[0]
        shape_str: str = first_row[f"{key}_shape"]
        dtype_str: str = first_row[f"{key}_dtype"]
        
        shape_list = shape_str.strip('()').split(',')
        shape_list = [s.strip() for s in shape_list if s.strip()]
        original_shape = tuple(map(int, shape_list))        
        dtype = np.dtype(dtype_str)
        
        num_rows = len(df_loaded)
        
        if len(original_shape) == 1:
            row_shape = ()
            full_shape = (num_rows,)
        else:
            row_shape = original_shape[1:]
            full_shape = (num_rows,) + row_shape
        
        restored_array = np.zeros(full_shape, dtype=dtype)
        
        for _, row in df_loaded.iterrows():
            row_index = row['row_index']
            data_val = row[f"{key}_data"]
            
            if row_shape == ():
                restored_array[row_index] = data_val[0]
            else:
                if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
                    row_data = np.array(data_val, dtype=dtype).reshape(row_shape)
                else:
                    row_data = np.frombuffer(data_val, dtype=dtype).reshape(row_shape)                
                restored_array[row_index] = row_data
        
        loaded_data[key] = restored_array

    for key in list_scalar_keys:
        loaded_data[key] = [0 for _ in range(len(df_loaded))]
        for _, row in df_loaded.iterrows():
            loaded_data[key][row['row_index']] = row[f"list_{key}"]
    
    for key in list_array_keys:
        first_row = df_loaded.iloc[0]
        dtype_str: str = first_row[f"list_{key}_dtype"]
        dtype = np.dtype(dtype_str)
        
        loaded_data[key] = [0 for _ in range(len(df_loaded))]
        
        for _, row in df_loaded.iterrows():
            row_index = row['row_index']
            data_val = row[f"list_{key}_data"]
            shape_str: str = row[f"list_{key}_shape"]
            shape_list = shape_str.strip('()').split(',')
            shape_list = [s.strip() for s in shape_list if s.strip()]
            original_shape = tuple(map(int, shape_list))  
            row_data = np.array(data_val, dtype=dtype).reshape(original_shape)
            loaded_data[key][row_index] = row_data
    
    return loaded_data
