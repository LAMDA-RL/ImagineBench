import abc
from typing import Tuple

import numpy as np
import gym

from clevr_robot_env import ClevrEnv
from utils.clevr_utils import CLEVR_QPOS_OBS_INDICES


class Simulator():
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    @abc.abstractmethod
    def set_state_from_obs(self, obs: np.ndarray) -> None:
        pass

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        self.set_state_from_obs(obs)
        return self.env.step(action)


class ClevrSimulator(Simulator):
    def __init__(self, env: ClevrEnv) -> None:
        super().__init__(env)

    def set_state_from_obs(self, obs: np.ndarray) -> None:
        assert obs.shape == 1

        # reset environment and get state
        self.env.reset()
        qpos, qvel = self.env.physics.data.qpos.copy(), self.env.physics.data.qvel.copy()

        # set state from obs
        qpos[CLEVR_QPOS_OBS_INDICES(self.env.num_object)] = obs
        self.env.set_state(qpos, qvel)

    def get_direct_obs(self) -> np.ndarray:
        return self.env.get_direct_obs()

    def get_image_obs(self) -> np.ndarray:
        return self.env.get_image_obs()

    def get_order_invariant_obs(self) -> np.ndarray:
        return self.env.get_order_invariant_obs()

    def get_obs(self) -> np.ndarray:
        return self.env.get_obs()

