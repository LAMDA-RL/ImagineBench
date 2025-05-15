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
import d3rlpy
import gymnasium
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from envs.mujoco import MujocoEnv
from d3rlpy.dataset.components import Episode
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.logging import TensorboardAdapterFactory
from d3rlpy.algos.qlearning import QLearningAlgoBase
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.dataset.types import Observation, ObservationSequence
from d3rlpy.dataset.mini_batch import TransitionMiniBatch, stack_observations, cast_recursively
from d3rlpy.dataset.transition_pickers import Transition, TransitionPickerProtocol, _validate_index, retrieve_observation, create_zero_observation


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


def convert_dataset(real_data, imagine_data, inst2enc) -> List:
    
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


        encoding = np.array([inst2enc[inst[0]] for inst in data['instructions']])
        encoding = encoding[:, np.newaxis, :].repeat(observations.shape[1], axis=1)

        observations = np.concatenate([observations, encoding], axis=-1)

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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_type", type=str, default="train", choices=['train', 'rephrase', 'easy', 'hard'], help="The type of offlineRL dataset.")
    parser.add_argument("--agent_name", type=str, default="bc", choices=['bc', 'cql', 'bcq', 'sac'], help="The name of offlineRL agent.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device for offlineRL training.")
    # offlineRL algorithm hyperparameters
    parser.add_argument("--seed", type=int, default=7, help="Seed.")

    parser.add_argument("--eval_episodes", type=int, default=10)

    args = parser.parse_args()

    return args


def EvalCallBackMujoco(agent: QLearningAlgoBase, epoch: int, total_step: int) -> None:

    if level == "train":
        env_dict = {"train": real_env, 
                    'rephrase': rephrase_env,
                    'easy': easy_env,
                    'hard': hard_env}
    elif level == 'rephrase':
        env_dict = {"train": real_env, 
                    'rephrase': rephrase_env}
    elif level == 'easy':
        env_dict = {'easy': easy_env}
    elif level == 'hard':
        env_dict = {'hard': hard_env}
    for test_level, env in env_dict.items():
        # assert isinstance(env.unwrapped, MyGrid)
        obs = env.reset()
        obs: np.ndarray

        traj_reward = 0
        traj_len = 0
        
        succ_list = []
        reward_list = []
        traj_len_list = []

        pbar = tqdm(total=args.eval_episodes, desc=f'epoch {epoch} eval: {test_level}', ncols=120, leave=False)
        while True:
            action = agent.predict(obs[np.newaxis, ...])
            action = action[0]
            obs, reward, done, info = env.step(action)
            traj_reward += reward
            traj_len += 1

            if done:
                succ_list.append(1 if info['success'] else 0)
                reward_list.append(traj_reward)
                traj_len_list.append(traj_len)
                
                pbar.update()

                if len(succ_list) == args.eval_episodes:
                    break
                else:
                    obs = env.reset()
                    traj_reward = 0
                    traj_len = 0
        pbar.close()

        
        agent.logger.add_metric(f'eval/{test_level}_succ', np.mean(succ_list))
        agent.logger.add_metric(f'eval/{test_level}_reward', np.mean(reward_list))
        agent.logger.add_metric(f'eval_len/{test_level}_len', np.mean(traj_len_list))
        
        if os.path.exists(json_log_path):
            with open(json_log_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data: Dict[str, Dict[int, float]]
        
        if test_level in data.keys():
            data[test_level][epoch] = np.mean(succ_list)
        else:
            data[test_level] = {epoch: np.mean(succ_list)}
        
        with open(json_log_path, 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    args = get_args()
    real_env = MujocoEnv(level = "real", dataset_url_dict={})
    rephrase_env = MujocoEnv(level = "rephrase", dataset_url_dict={})
    easy_env = MujocoEnv(level = "easy", dataset_url_dict={})
    hard_env = MujocoEnv(level = "hard", dataset_url_dict={})
    level = args.ds_type
    if level == "train":
        env = MujocoEnv(level = "real", dataset_url_dict={})
    else:
        env = MujocoEnv(level = level, dataset_url_dict={})
    inst2encode = env.inst2encode
    real_dataset_path = "../../rimro_offline/data/data/mujoco_real.npy"
    real_dataset = np.load(real_dataset_path, allow_pickle=True).item()
    if level == "train":
        imagine_dataset = None
    else:
        imagine_dataset_path = f"../../rimro_offline/data/data/mujoco_{level}.npy"
        imagine_dataset = np.load(imagine_dataset_path, allow_pickle=True).item()
    llata_episodes = convert_dataset(
        real_data=real_dataset, imagine_data=imagine_dataset,
        inst2enc = inst2encode
        )

    dataset = LlataReplayBuffer(
        InfiniteBuffer(),
        transition_picker=LlataTransitionPicker(),
        episodes=llata_episodes,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.agent_name == 'bc':
        agent = d3rlpy.algos.BCConfig(
            encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'cql':
        agent = d3rlpy.algos.CQLConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'bcq':
        agent = d3rlpy.algos.BCQConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'td3+bc':
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.TD3PlusBCConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    else:
        raise NotImplementedError
    exp_name = 'Mujoco'
    kwargs = vars(args)    
    exp_name = f'{exp_name}_{"i-" if imagine_dataset is not None else ""}{kwargs["ds_type"].replace("_level", "")}'
    exp_name_temp = f'{exp_name}_{kwargs["agent_name"]}_seed{kwargs["seed"]}'
    exp_name = f'{exp_name_temp}_{datetime.now().strftime("%m-%d_%H-%M-%S")}'

    exp_num = 0
    json_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "json_log", f"exp_{exp_num}")
    os.makedirs(json_log_dir, exist_ok=True)
    json_log_path = os.path.join(json_log_dir, f'{exp_name_temp}.json')

    # offline training
    agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name=exp_name,
        with_timestamp=False,
        logger_adapter=TensorboardAdapterFactory(root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        epoch_callback=EvalCallBackMujoco,
    )

    print("===> offline training finished")
