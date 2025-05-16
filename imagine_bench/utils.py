import os
import json
import random
import argparse
import dataclasses
from datetime import datetime
from typing import Any, Dict, List, Sequence, Union
import sys
import h5py
import torch
import algo.d3rlpy as d3rlpy
import gymnasium
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from envs.mujoco import MujocoEnv
from algo.d3rlpy.dataset.components import Episode
from algo.d3rlpy.models.encoders import EncoderFactory
from algo.d3rlpy.logging import TensorboardAdapterFactory
from algo.d3rlpy.algos.qlearning import QLearningAlgoBase
from algo.d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from algo.d3rlpy.dataset.types import Observation, ObservationSequence
from algo.d3rlpy.dataset.mini_batch import TransitionMiniBatch, stack_observations, cast_recursively
from algo.d3rlpy.dataset.transition_pickers import Transition, TransitionPickerProtocol, _validate_index, retrieve_observation, create_zero_observation


@dataclasses.dataclass(frozen=True)
class LlataEpisode(Episode):
    observations: ObservationSequence
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool

    def serialize(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
        }

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "LlataEpisode":
        return cls(
            observations=serializedData["observations"],
            actions=serializedData["actions"],
            rewards=serializedData["rewards"],
            terminated=serializedData["terminated"],
        )


def convert_dataset(real_data, imagine_data) -> List:
    
    llata_episodes = []
    for data in [real_data, imagine_data]:
        if data is None:
            continue

        valid_len_arr = np.array(data["masks"]).sum(axis=1)
        valid_len_arr: np.ndarray = valid_len_arr.astype(int)
        observations = np.array(data['observations'])
        actions = np.array(data['actions'])
        if len(actions.shape) == 2:
            actions = actions[..., np.newaxis]
        rewards = np.array(data['rewards'])

        for tau_idx in tqdm(range(valid_len_arr.shape[0])):
            tau_len = valid_len_arr[tau_idx].item()
            tau_obs_arr = observations[tau_idx][:tau_len]
            tau_action_arr = actions[tau_idx][:tau_len]
            tau_reward_arr = rewards[tau_idx][:tau_len]

            llata_episode = LlataEpisode(observations=tau_obs_arr, actions=tau_action_arr, rewards=tau_reward_arr, terminated=True)
            llata_episodes.append(llata_episode)

    return llata_episodes


class LlataEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size, hidden_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_dim = feature_size
        
        self.fc1 = nn.Linear(observation_shape[0], self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.feature_size)
    
    def get_feature_size(self):
        return self.feature_size

    def forward(self, x):
        assert len(x.shape) == 2, 'x must be 2-dim tensor. (batch_size, observation_size)'

        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))

        return h


# your own encoder factory
@dataclasses.dataclass()
class LlataEncoderFactory(EncoderFactory):
    feature_size: int
    hidden_size: int

    def create(self, observation_shape):
        return LlataEncoder(observation_shape, self.feature_size, self.hidden_size)

    @staticmethod
    def get_type() -> str:
        return "Llata"


@dataclasses.dataclass(frozen=True)
class LlataTransition(Transition):
    observation: Observation  # (...)
    action: np.ndarray  # (...)
    reward: np.ndarray  # (1,)
    next_observation: Observation  # (...)
    terminal: float
    interval: int


class LlataTransitionPicker(TransitionPickerProtocol):
    def __call__(self, episode: LlataEpisode, index: int) -> LlataTransition:
        _validate_index(episode, index)

        observation = retrieve_observation(episode.observations, index)
        is_terminal = episode.terminated and index == episode.size() - 1
        if is_terminal:
            next_observation = create_zero_observation(observation)
        else:
            next_observation = retrieve_observation(
                episode.observations, index + 1
            )
        return LlataTransition(
            observation=observation,
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=next_observation,
            terminal=float(is_terminal),
            interval=1,
        )


@dataclasses.dataclass(frozen=True)
class LlataTransitionMiniBatch(TransitionMiniBatch):
    observations: Union[np.ndarray, Sequence[np.ndarray], Dict]  # (B, ...)
    actions: np.ndarray  # (B, ...)
    rewards: np.ndarray  # (B, 1)
    next_observations: Union[np.ndarray, Sequence[np.ndarray], Dict]  # (B, ...)
    terminals: np.ndarray  # (B, 1)
    intervals: np.ndarray  # (B, 1)

    @classmethod
    def from_transitions(
        cls, transitions: Sequence[Transition]
    ) -> "LlataTransitionMiniBatch":
        observations = stack_observations(
            [transition.observation for transition in transitions]
        )
        actions = np.stack(
            [transition.action for transition in transitions], axis=0
        )
        rewards = np.stack(
            [transition.reward for transition in transitions], axis=0
        )
        next_observations = stack_observations(
            [transition.next_observation for transition in transitions]
        )
        terminals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        intervals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )

        return LlataTransitionMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
        )


class LlataReplayBuffer(ReplayBuffer):
    def sample_transition_batch(self, batch_size: int) -> LlataTransitionMiniBatch:
        r"""Samples a mini-batch of transitions.

        Args:
            batch_size: Mini-batch size.

        Returns:
            Mini-batch.
        """
        return LlataTransitionMiniBatch.from_transitions(
            [self.sample_transition() for _ in range(batch_size)]
        )
    
def make_d3rlpy_dataset(real_data, imagine_data):
    llata_episodes = convert_dataset(
        real_data=real_data, imagine_data=imagine_data,
        )
    dataset = LlataReplayBuffer(
            InfiniteBuffer(),
            transition_picker=LlataTransitionPicker(),
            episodes=llata_episodes,
        )
    return dataset