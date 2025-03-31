import sys
import h5py
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import numpy as np
from envs import RIMAROEnv
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
        self.max_episode_steps = 50
        self.num_object = 5
        self.level = kwargs['level']
        self.hist_obs_list = None
        self.timestep = 0
        self.ptr = None

        # 指令以及其余信息统一存为 npy, 数据集(obs, action, mask, reward)统一存为 h5
        self.real_npy_path = Path(__file__).parent.parent.joinpath('data/ball_real.npy')
        self.real_h5_path = Path(__file__).parent.parent.joinpath('data/ball_real.h5')

        self.imaginary_easy_npy_path = Path(__file__).parent.parent.joinpath('data/ball_imaginary_easy.npy')
        self.imaginary_easy_h5_path = Path(__file__).parent.parent.joinpath('data/ball_imaginary_easy.h5')
        self.imaginary_rephrase_npy_path = Path(__file__).parent.parent.joinpath('data/ball_imaginary_rephrase.npy')
        self.imaginary_rephrase_h5_path = Path(__file__).parent.parent.joinpath('data/ball_imaginary_rephrase.h5')
        self.imaginary_hard_npy_path = Path(__file__).parent.parent.joinpath('data/ball_imaginary_hard.npy')
        self.imaginary_hard_h5_path = Path(__file__).parent.parent.joinpath('data/ball_imaginary_hard.h5')

        real_data = np.load(self.real_npy_path, allow_pickle=True).item()
        self.real_data_info = {
            'instructions': real_data['instructions'],
            'goals': real_data['goals'],
        }
        self.imaginary_data_info = {}
        imaginary_easy_data = np.load(self.imaginary_easy_npy_path, allow_pickle=True).item()
        self.imaginary_data_info['easy_level'] = {
            'instructions': imaginary_easy_data['instructions'],
            'goals': imaginary_easy_data['goals'],
        }
        imaginary_rephrase_data = np.load(self.imaginary_rephrase_npy_path, allow_pickle=True).item()
        self.imaginary_data_info['rephrase_level'] = {
            'instructions': imaginary_rephrase_data['instructions'],
            'goals': imaginary_rephrase_data['goals'],
        }
        imaginary_hard_data = np.load(self.imaginary_hard_npy_path, allow_pickle=True).item()
        self.imaginary_data_info['hard_level'] = {
            'instructions': imaginary_hard_data['instructions'],
            'goals': imaginary_hard_data['goals'],
        }
        super().__init__(
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

        obs = super().reset(**kwargs)
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
        next_obs, reward, done, info = super().step(action)

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
    
    def get_dataset(self):
        real_data_info = {}
        with h5py.File(self.real_h5_path, "r") as real_fr:
            real_data_info['masks'] = np.array(real_fr["masks"])
            real_data_info['observations'] = np.array(real_fr['observations'])
            real_data_info['actions'] = np.array(real_fr['actions'])
            real_data_info['rewards'] = np.array(real_fr['rewards'])
        imaginary_data_info = {}
        imaginary_data_info['easy_level'] = {}
        with h5py.File(self.imaginary_easy_h5_path, "r") as imaginary_fr:
            imaginary_data_info['easy_level']['masks'] = np.array(imaginary_fr["masks"])
            imaginary_data_info['easy_level']['observations'] = np.array(imaginary_fr['observations'])
            imaginary_data_info['easy_level']['actions'] = np.array(imaginary_fr['actions'])
            imaginary_data_info['easy_level']['rewards'] = np.array(imaginary_fr['rewards'])
        imaginary_data_info['rephrase_level'] = {}
        with h5py.File(self.imaginary_rephrase_h5_path, "r") as imaginary_fr:
            imaginary_data_info['rephrase_level']['masks'] = np.array(imaginary_fr["masks"])
            imaginary_data_info['rephrase_level']['observations'] = np.array(imaginary_fr['observations'])
            imaginary_data_info['rephrase_level']['actions'] = np.array(imaginary_fr['actions'])
            imaginary_data_info['rephrase_level']['rewards'] = np.array(imaginary_fr['rewards'])
        imaginary_data_info['hard_level'] = {}
        with h5py.File(self.imaginary_hard_h5_path, "r") as imaginary_fr:
            imaginary_data_info['hard_level']['masks'] = np.array(imaginary_fr["masks"])
            imaginary_data_info['hard_level']['observations'] = np.array(imaginary_fr['observations'])
            imaginary_data_info['hard_level']['actions'] = np.array(imaginary_fr['actions'])
            imaginary_data_info['hard_level']['rewards'] = np.array(imaginary_fr['rewards'])

        dataset_info = {
            'real_data': real_data_info,
            'imaginary_data': imaginary_data_info,
        }

        return dataset_info
    
    def get_instruction(self):
        # 存在多条语义相同指令
        instruction = np.random.choice(self.real_data_info['instructions'][self.ptr])

        return instruction
