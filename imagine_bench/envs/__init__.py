import os
import h5py
import urllib.request
from pathlib import Path
import progressbar
import urllib.request
from gymnasium import spaces

DATASET_PATH = Path(__file__).parent.joinpath('data')
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
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
        
    def init_dataset(self):
        raise NotImplementedError

    def get_dataset(self, **kwargs):
        raise NotImplementedError
    
    def get_instruction(self):
        raise NotImplementedError


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


url2ds_name = {
    # ball
    'https://box.nju.edu.cn/f/c67fb5ed23694db0baaa/?dl=1': 'ball_imaginary_easy.h5',
    'https://box.nju.edu.cn/f/e857ae10a53a4758a81f/?dl=1': 'ball_imaginary_easy.npy',
    'https://box.nju.edu.cn/f/ac414bd4cf014dce87c6/?dl=1': 'ball_imaginary_hard.h5',
    'https://box.nju.edu.cn/f/53e31be7ab9248e4b292/?dl=1': 'ball_imaginary_hard.npy',
    'https://box.nju.edu.cn/f/1185b8b8673a47daae6c/?dl=1': 'ball_imaginary_rephrase.h5',
    'https://box.nju.edu.cn/f/d3bceb9d5c5248d8b410/?dl=1': 'ball_imaginary_rephrase.npy',
    'https://box.nju.edu.cn/f/76fd601e30e54975809f/?dl=1': 'ball_real.h5',
    'https://box.nju.edu.cn/f/ede29bd4d9d74c93a6d4/?dl=1': 'ball_real.npy',
    # metaworld
    'https://box.nju.edu.cn/f/6b56624598f0487fb65a/?dl=1': 'metaworld_imaginary_easy.h5',
    'https://box.nju.edu.cn/f/d2f9ec3ddf8c46c6a2e3/?dl=1': 'metaworld_imaginary_hard.h5',
    'https://box.nju.edu.cn/f/5515ee501db948eb84db/?dl=1': 'metaworld_imaginary_rephrase.h5',
    'https://box.nju.edu.cn/f/0fea8e468869468092e9/?dl=1': 'metaworld_real.h5',
    # babyai
    'https://box.nju.edu.cn/f/da1fc389e5d24c45a3a4/?dl=1': 'babyai_imaginary_easy.npy',
    'https://box.nju.edu.cn/f/b8c6282e2ddf4819b972/?dl=1': 'babyai_imaginary_hard.npy',
    'https://box.nju.edu.cn/f/e4d6695bcbe141bf927b/?dl=1': 'babyai_imaginary_rephrase.npy',
    'https://box.nju.edu.cn/f/47ef43a660874409a420/?dl=1': 'babyai_real.npy',
    # libero
    'https://box.nju.edu.cn/f/c8a4efcff22c46098535/?dl=1': 'libero_imaginary_easy.npy',
    'https://box.nju.edu.cn/f/39741deb735843ddbeec/?dl=1': 'libero_imaginary_hard.npy',
    'https://box.nju.edu.cn/f/e1f3d5d9c7bb4fc78a2d/?dl=1': 'libero_imaginary_rephrase.npy',
    'https://box.nju.edu.cn/f/93ef16b8e2d64f5ea935/?dl=1': 'libero_real.npy',
    # mujoco
    'https://box.nju.edu.cn/f/dbf3a096b380460bb1b9/?dl=1': 'mujoco_imaginary_easy.npy',
    'https://box.nju.edu.cn/f/f70e67f27c8d40d59568/?dl=1': 'mujoco_imaginary_hard.npy',
    'https://box.nju.edu.cn/f/c53bc2307c6d49ea9b0b/?dl=1':'mujoco_imaginary_easy.npy',
    'https://box.nju.edu.cn/f/44f5c558982e4dc3b5d5/?dl=1':'mujoco_real.npy',
}


def filepath_from_url(dataset_url):
    dataset_name = url2ds_name[dataset_url]
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


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath, show_progress)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    
    return dataset_filepath


# 实现类似 gym 的 make 函数
def make(env_id: str, **kwargs) -> RIMAROEnv:
    if env_id not in ENV_ID_LIST:
        raise ValueError(f"env_id {env_id} is not supported.")
    
    env = None
    kwargs['level'] = kwargs.get('level', 'real')
    if env_id == 'Ball-v0':
        from envs.ball import BallEnv
        kwargs['dataset_url_dict'] = {
            'imaginary_rephrase_h5': 'https://box.nju.edu.cn/f/c67fb5ed23694db0baaa/?dl=1',
            'imaginary_easy_npy': 'https://box.nju.edu.cn/f/e857ae10a53a4758a81f/?dl=1',
            'imaginary_hard_h5': 'https://box.nju.edu.cn/f/ac414bd4cf014dce87c6/?dl=1',
            'imaginary_hard_npy': 'https://box.nju.edu.cn/f/53e31be7ab9248e4b292/?dl=1',
            'imaginary_easy_h5': 'https://box.nju.edu.cn/f/1185b8b8673a47daae6c/?dl=1',
            'imaginary_rephrase_npy': 'https://box.nju.edu.cn/f/d3bceb9d5c5248d8b410/?dl=1',
            'real_h5': 'https://box.nju.edu.cn/f/76fd601e30e54975809f/?dl=1',
            'real_npy': 'https://box.nju.edu.cn/f/ede29bd4d9d74c93a6d4/?dl=1',
        }
        env = BallEnv(**kwargs)
    elif env_id == 'MetaWorld-v0':
        from envs.metaworld import MetaWorldEnv
        kwargs['dataset_url_dict'] = {
            'imaginary_easy_h5': 'https://box.nju.edu.cn/f/6b56624598f0487fb65a/?dl=1',
            'imaginary_hard_h5': 'https://box.nju.edu.cn/f/d2f9ec3ddf8c46c6a2e3/?dl=1',
            'imaginary_rephrase_h5': 'https://box.nju.edu.cn/f/5515ee501db948eb84db/?dl=1',
            'real_h5': 'https://box.nju.edu.cn/f/0fea8e468869468092e9/?dl=1',
        }
        env = MetaWorldEnv(**kwargs)
    elif env_id == 'BabyAI-v0':
        from envs.babyai import BabyAIEnv
        kwargs['dataset_url_dict'] = {
            'easy': 'https://box.nju.edu.cn/f/da1fc389e5d24c45a3a4/?dl=1',
            'hard': 'https://box.nju.edu.cn/f/b8c6282e2ddf4819b972/?dl=1',
            'rephrase': 'https://box.nju.edu.cn/f/e4d6695bcbe141bf927b/?dl=1',
            'real': 'https://box.nju.edu.cn/f/47ef43a660874409a420/?dl=1',
        }
        env = BabyAIEnv(**kwargs)
    elif env_id == 'Libero-v0':
        from envs.libero import LiberoEnv
        kwargs['dataset_url_dict'] = {
            'easy': 'https://box.nju.edu.cn/f/c8a4efcff22c46098535/?dl=1',
            'hard': 'https://box.nju.edu.cn/f/39741deb735843ddbeec/?dl=1',
            'rephrase': 'https://box.nju.edu.cn/f/e1f3d5d9c7bb4fc78a2d/?dl=1',
            'real': 'https://box.nju.edu.cn/f/93ef16b8e2d64f5ea935/?dl=1',
        }
        env = LiberoEnv(**kwargs)
    elif env_id == 'Mujoco-v0':
        from envs.mujoco import MujocoEnv
        kwargs['dataset_url_dict'] = {
            'easy': 'https://box.nju.edu.cn/f/dbf3a096b380460bb1b9/?dl=1',
            'hard': 'https://box.nju.edu.cn/f/f70e67f27c8d40d59568/?dl=1',
            'rephrase': 'https://box.nju.edu.cn/f/c53bc2307c6d49ea9b0b/?dl=1',
            'real': 'https://box.nju.edu.cn/f/44f5c558982e4dc3b5d5/?dl=1',
        }
        env = MujocoEnv(**kwargs)
    else:
        raise NotImplementedError
    
    return env
