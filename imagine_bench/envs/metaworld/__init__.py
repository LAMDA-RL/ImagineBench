"""Proposal for a simple, understandable MetaWorld API."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import abc
import pickle
from collections import OrderedDict
from typing import List, NamedTuple, Type

import numpy as np

import metaworld.envs.mujoco.env_dict as _env_dict
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

EnvName = str


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: EnvName
    data: bytes  # Contains env parameters like random_init and *a* goal


# class MetaWorldEnv:
#     """Environment that requires a task before use.

#     Takes no arguments to its constructor, and raises an exception if used
#     before `set_task` is called.
#     """

#     def set_task(self, task: Task) -> None:
#         """Set the task.

#         Raises:
#             ValueError: If task.env_name is different from the current task.

#         """


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> "OrderedDict[EnvName, Type]":
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> "OrderedDict[EnvName, Type]":
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
_MT_OVERRIDE = dict(partially_observable=False)

_N_GOALS = 50


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, kwargs_override, seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    tasks = []
    for env_name, args in args_kwargs.items():
        assert len(args["args"]) == 0
        env = classes[env_name]()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args["kwargs"].copy()
        del kwargs["task_id"]
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS, unique_task_rand_vecs.shape[
            0
        ]
        env.close()
        for rand_vec in rand_vecs:
            kwargs = args["kwargs"].copy()
            del kwargs["task_id"]
            kwargs.update(dict(rand_vec=rand_vec, env_cls=classes[env_name]))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
        del env
    if seed is not None:
        np.random.set_state(st0)
    return tasks


def _ml1_env_names():
    tasks = list(_env_dict.ML1_V2["train"])
    assert len(tasks) == 50 + 2 + 10  # 12 newly added task
    return tasks


class ML1(Benchmark):
    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes,
            {env_name: args_kwargs},
            _ML_OVERRIDE,
            seed=(seed + 1 if seed is not None else seed),
        )


class MT1(Benchmark):
    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []


class ML10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML10_V2["train"]
        self._test_classes = _env_dict.ML10_V2["test"]
        train_kwargs = _env_dict.ml10_train_args_kwargs

        test_kwargs = _env_dict.ml10_test_args_kwargs
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )

        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


class ML45(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML45_V2["train"]
        self._test_classes = _env_dict.ML45_V2["test"]
        train_kwargs = _env_dict.ml45_train_args_kwargs
        test_kwargs = _env_dict.ml45_test_args_kwargs

        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


class MT10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT10_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT10_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []
        self._test_classes = []


class MT50(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT50_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT50_V2_ARGS_KWARGS

        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []


__all__ = ["ML1", "MT1", "ML10", "MT10", "ML45", "MT50"]


level2true_level = {
    'real': 'baseline',
    'rephrase': 'rephrase_level',
    'easy': 'easy_level',
    'hard': 'hard_level',
}


import h5py
from gymnasium import spaces
from metaworld.meta_utils import MetaWrapper, baseline_env_name_list, rephrase_level_env_name_list, easy_level_env_name_list, hard_level_env_name_list
from metaworld.meta_utils import TAU_LEN, num_nl, data_dir, wrap_info, en2nl, obs_online2noisy_offline, get_noisy_entity_list, num_noisy_entity
from envs import LEVEL_LIST, download_dataset_from_url
class MetaWorldEnv:
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
            eval_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
            eval_env._freeze_rand_vec = False  # random reset
            eval_env.max_path_length = TAU_LEN  # Try to avoid returns of successful tau less than failed tau
            eval_env = MetaWrapper(eval_env, wrap_info=wrap_info)
            self.env_list.append(eval_env)

        self.inst_encoding_dict = np.load(data_dir.joinpath('meta_instructions_encoding.npy'), allow_pickle=True).item()
        self.ptr = None
        self.inst_ptr = None
        self.tau_noisy_entity_list = None

        assert len(self.env_list) > 0
        online_obs, _ = self.env_list[0].reset()
        noisy_entity_list = get_noisy_entity_list(env_name=self.env_name_list[0])
        tau_noisy_entity_list = np.random.choice(noisy_entity_list, size=num_noisy_entity)
        offline_obs = obs_online2noisy_offline(self.env_name_list[0], online_obs, tau_noisy_entity_list=tau_noisy_entity_list)
        env_obs = offline_obs.copy()
        inst_encoding = self.inst_encoding_dict[self.env_name_list[0]][0].flatten()
        policy_obs = np.r_[env_obs, inst_encoding]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=policy_obs.shape, dtype=np.float32)
        self.action_space = self.env_list[0].action_space
        
        # 指令可以基于 env_name 获取, 数据集(obs, action, mask, reward)统一存为 h5
        self.path_dict = {}
        # self.init_dataset()
    
    def get_policy_obs(self, obs: np.ndarray):
        env_name = self.env_name_list[self.ptr]

        offline_obs = obs_online2noisy_offline(env_name, obs, tau_noisy_entity_list=self.tau_noisy_entity_list)

        env_obs = offline_obs.copy()
        inst_encoding = self.inst_encoding_dict[env_name][self.inst_ptr].flatten()
        policy_obs = np.r_[env_obs, inst_encoding]

        return policy_obs

    def reset(self, **kwargs):
        if self.ptr is None:
            self.ptr = 0
        else:
            self.ptr = (self.ptr + 1) % len(self.env_list)
        self.inst_ptr = np.random.randint(0, num_nl)
        
        env_name = self.env_name_list[self.ptr]
        noisy_entity_list = get_noisy_entity_list(env_name=env_name)
        tau_noisy_entity_list = np.random.choice(noisy_entity_list, size=num_noisy_entity)
        self.tau_noisy_entity_list = tau_noisy_entity_list

        curr_env = self.env_list[self.ptr]

        obs, info = curr_env.reset(**kwargs)

        return self.get_policy_obs(obs=obs), info

    def step(self, action):
        curr_env = self.env_list[self.ptr]

        next_obs, reward, terminated, truncated, info = curr_env.step(action)
        terminated = terminated or bool(info['success'])

        return self.get_policy_obs(obs=next_obs), reward, terminated, truncated, info
     
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
        env_name = self.env_name_list[self.ptr]

        instruction = en2nl[env_name][self.inst_ptr]

        return instruction
