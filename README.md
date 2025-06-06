<h1>ImagineBench: Evaluating Reinforcement Learning
with Large Language Model Rollouts</h1>

[![Project Status: Active](https://img.shields.io/badge/status-active-green)](https://github.com/LAMDA-RL/ImagineBench)

## Overview

![Overview of ImagineBench](./docs/overview.png)

A benchmark for evaluating reinforcement learning algorithms that train the policies using both **real data** and **imaginary rollouts from LLMs**. The concept of imaginary rollouts was proposed by [KALM](https://openreview.net/forum?id=tb1MlJCY5g) (NeurIPS 2024), which focuses on extracting knowledge from LLMs, in the form of environmental rollouts, to improve RL policies' performance on novel tasks. 
Please check [the paper for ImagineBench](https://arxiv.org/abs/2505.10010v1) for more details.

**Core focus**: Measuring how well agents can learn effective policies through LLM's imaginary rollouts and generalize well on novel tasks.


## 📢 News
- **May 15, 2025**: The paper about ImagineBench is accessible at [arXiv](https://arxiv.org/abs/2505.10010v1).
- **May 14, 2025**: Add MuJoCo (HalfCheetah) environment to the benchmark, focusing on robotics locomotion.
- **Apr 3, 2025**: Add BabyAI and LIBERO environments to the benchmark.
- **Mar 31, 2025**: Initial release of datasets for CLEVR-Robot and Meta-World and the environments with Gymnasium wrapper.


## Dataset Status


We have released initial datasets for diverse environments, with both real+LLM-generated rollouts.
More environments and tasks are under active development.

### Available Environments
| Environment | Training tasks                                                  | Novel tasks                                                                                        |
|-------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| CLEVR-Robot | move A ball to one direction of B ball                          | unseen tasks such as "gather all the balls together" and "arrange the balls in a line"             |
| Meta-World  | ten different tasks provided by Meta-world benchmark            | manipulation under the assumption that the agent is facing a wall / combination of training skills |
| BabyAI      | 7x7 Grid world with task like "goto", "putnext" and "open the door" | novel combination and extension of the training skills                                             |
| LIBERO      | robotic manipulation involving pick and place                   | sequential pick and place / manipulation under the assumption of unsafe factors                    |
| MuJoCo      | robotic locomotion involving running forward/backward    | sequential run forward and backward / jump in place                                                |

---

We are actively preparing:

- More environment domains
- Real+LLM-imaginary rollouts
- Varying task difficulty levels

## Installtion

Please run the following commands in the given order to install the dependency for **ImagineBench**.

```
conda create -n imagine_bench python=3.10.13
conda activate imagine_bench
git clone git@github.com:LAMDA-RL/ImagineBench.git
cd ImagineBench
pip install -r requirements.txt
```
Then install the ImagineBench package:
```
pip install -e .
```
## Data in ImagineBench
In ImagineBench, real- and imaginary-datasets returned by `get_dataset()` function are in `dict` type with the same format, where **N** is the number of rollouts and **T** is max trajectory length.

- `observations`: An **(N, T, D)** array, where *D* is dim of observation space concatenated with instruction encoding.

- `actions`: An **(N, T, D)** array, where *D* is dim of action space.

- `rewards`: An **(N, T, 1)** array.

- `masks`: An **(N, T, 1)** array indicating whether each time step in a trajectory is valid(1) or padding(0).

## Basic usage

**Offline RL Training with one line**

To start offline RL training with imaginary rollouts, you can set:

`ALGO` from `[bc, cql, bcq, td3+bc]`

`ENV` from `[Ball-v0, MetaWorld-v0, BabyAI-v0, Libero-v0, Mujoco-v0]`

`DATASET_TYPE` from `[train, rephrase, easy, hard]`

```
python imagine_bench/train.py --algo ALGO --env ENV --ds_type DATASET_TYPE  --device DEVICE --seed SEED --eval_episodes 40
```

**Get dataset** 
```python
import imagine_bench

# Optional task_level: ['real', 'rephrase', 'easy', 'hard'].
env = imagine_bench.make('MetaWorld-v0', level='rephrase')
real_data, imaginary_rollout_rephrase = env.get_dataset(level="rephrase") 

# Or you can use the dataset with other task levels.
env = imagine_bench.make('MetaWorld-v0', level='easy')
real_data, imaginary_rollout_easy = env.get_dataset(level="easy")
```

**Example for Offline RL Training with [d3rlpy](https://github.com/takuseno/d3rlpy) Repo** 
```python
import algo.d3rlpy as d3rlpy
from algo.d3rlpy.logging import TensorboardAdapterFactory
import os
import imagine_bench
from imagine_bench.utils import LlataEncoderFactory, make_d3rlpy_dataset
from evaluations import CallBack

env = imagine_bench.make('Mujoco-v0', level='rephrase')
real_data, imaginary_rollout_rephrase = env.get_dataset(level="rephrase") 
dataset = make_d3rlpy_dataset(real_data, imaginary_rollout_rephrase)

agent = d3rlpy.algos.BCConfig(
                encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            ).create(device="cuda:0")
callback = CallBack()
callback.add_eval_env(env_dict={"train": env}, eval_num=10)
agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name="mujoco",
        epoch_callback=callback.EvalCallback,
        logger_adapter=TensorboardAdapterFactory(root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
```

**Example for Offline RL Training with [OfflineRL](https://github.com/polixir/OfflineRL) Repo** 
```python
import random
import argparse
from datetime import datetime

import torch
import numpy as np

import imagine_bench
from utils import LlataEncoderFactory, make_offlinerl_dataset
from gym import spaces
from algo.offlinerl.algo import algo_select
from evaluations import EvalCallBackFunction
from torch.utils.tensorboard import SummaryWriter

env = imagine_bench.make('Mujoco-v0', level='rephrase')
real_data, imaginary_rollout_rephrase = env.get_dataset(level="rephrase") 
dataset, data_num = make_offlinerl_dataset(real_data, imaginary_rollout_rephrase)

algo_args = {
    "algo_name": 'bc',
    "seed": 0,
    "obs_shape": env.observation_space.shape[0],
    "action_shape": env.action_space.shape[0],
    "device": "cuda:0",
    "exp_name": f"example",
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "max_action": float(spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32).high[0]),
    "task": 'mujoco'
}
writer  = SummaryWriter(f"logs")
algo_init_fn, algo_trainer_obj, algo_config = algo_select(algo_args)
algo_config['data_name'] = f"mujoco_rephrase"
algo_init = algo_init_fn(algo_config)
algo_trainer = algo_trainer_obj(algo_init, algo_config)
callback = EvalCallBackFunction()
callback.initialize({'train':env}, 10, writer)
algo_trainer.train(train_buffer=dataset, val_buffer=None, callback_fn=callback)
```


## Citation

Please cite ImagineBench if you find this benchmark useful in your research:

```
@article{pang2025imaginebench,
  title={ImagineBench: Evaluating Reinforcement Learning with Large Language Model Rollouts},
  author={Jing-Cheng Pang and
          Kaiyuan Li and
          Yidi Wang and
          Si-Hang Yang and 
          Shengyi Jiang and 
          Yang Yu},
  journal={arXiv preprint arXiv:2505.10010},
  year={2025}
}
```