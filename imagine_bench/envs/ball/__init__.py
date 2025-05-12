import sys
import h5py
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import numpy as np
from envs import LEVEL_LIST, RIMAROEnv, download_dataset_from_url
from clevr_robot_env import LlataEnv
from utils.clevr_utils import terminal_fn_with_level, CLEVR_QPOS_OBS_INDICES


level2true_level = {
    'real': 'tau_level',
    'rephrase': 'tau_level',
    'easy': 'step_level',
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
        
        self.path_dict['real_npy'] = download_dataset_from_url(self.dataset_url_dict['real_npy'])
        real_data = np.load(self.path_dict['real_npy'], allow_pickle=True).item()
        self.real_data_info = {
            'instructions': real_data['instructions'],
            'goals': real_data['goals'],
        }
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
        
    def reset(self, **kwargs):
        self.level = kwargs.get('level', 'real')

        obs = LlataEnv.reset(self, **kwargs)
        info = {}

        qpos = self.physics.data.qpos.copy()
        init_env_obs = qpos[CLEVR_QPOS_OBS_INDICES(self.num_object)]
        self.hist_obs_list = [init_env_obs]

        if self.ptr is None:
            self.ptr = 0
        else:
            self.ptr = (self.ptr + 1) % len(self.real_data_info['goals'])
        self.goal_arr = self.real_data_info['goals'][self.ptr]
        self.timestep = 0

        return obs, info

    def step(self, action):
        next_obs, reward, done, info = LlataEnv.step(self, action)

        true_level = level2true_level[self.level]
        terminal_kwargs = dict(
            insts=None,
            observations=np.array([next_obs]),
            number_of_objects=self.num_object,
            goals=np.array([self.goal_arr]),
            level=true_level,
        )
        if self.hist_obs_list is None:
            eval_result = {
                'done': False,
            }
        else:
            if 'step_level' in true_level:
                terminal_kwargs['hist_observations'] = np.array(self.hist_obs_list).reshape(1, 2, 2 * self.num_object)
                terminal_kwargs['actions'] = np.array([action]).reshape(1, -1)
            eval_result = terminal_fn_with_level(**terminal_kwargs)
            qpos = self.physics.data.qpos.copy()
            env_obs = qpos[CLEVR_QPOS_OBS_INDICES(self.num_object)]
            self.hist_obs_list.append(env_obs)

        self.timestep += 1

        terminated = eval_result['done']
        trunated = self.timestep >= self.max_episode_steps

        return next_obs, reward, terminated, trunated, info
    
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
        instruction = np.random.choice(self.real_data_info['instructions'][self.ptr])

        return instruction
