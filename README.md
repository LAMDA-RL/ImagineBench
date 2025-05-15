<h1>ImagineBench: Evaluating Reinforcement Learning
with Large Language Model Rollouts</h1>

[![Project Status: Active](https://img.shields.io/badge/status-active-green)](https://github.com/LAMDA-RL/ImagineBench)

## Overview

A benchmark for evaluating reinforcement learning algorithms that train the policies using both **real data** and **imaginary rollouts from LLMs**. The concept of imaginary rollouts was proposed by [KALM](https://openreview.net/forum?id=tb1MlJCY5g) (NeurIPS 2024), which focuses on extracting knowledge from LLMs, in the form of environmental rollouts, to improve RL policies' performance on novel tasks. Please check the paper for more details.

**Core focus**: Measuring how well agents can learn effective policies through LLM's imaginary rollouts and generalize well on novel tasks.


## 📢 News
- **Apr 3, 2025**: Add BabyAI and LIBERO environments to the benchmark.
- **Mar 31, 2025**: Initial release of datasets for CLEVR-Robot and Meta-World and the environments with Gymnasium wrapper.


## Dataset Status

**Now Available!**  

We have released initial datasets for 4 environments: [CLEVR-Robot](https://github.com/google-research/clevr_robot_env), [Meta-World](https://github.com/Farama-Foundation/Metaworld), [BabyAI](https://github.com/mila-iqia/babyai) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) with both real+LLM-generated rollouts.
More environments and tasks are under active development.

### Available Environments
| Environment | Training tasks                                                      | Novel tasks                                                                                        | LLM Sources |
|-------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|-------------|
| CLEVR-Robot | move A ball to one direction of B ball                              | unseen tasks such as "gather all the balls together" and "arrange the balls in a line"             | llama2      |
| Meta-World  | ten different tasks provided by Meta-world benchmark                | manipulation under the assumption that the agent is facing a wall / combination of training skills | llama2      |
| BabyAI      | 7x7 Grid world with task like "goto", "putnext" and "open the door" | novel combination and extension of the training skills                                             | llama2      |
| LIBERO      | robotic manipulation involving pick and place                       | sequential pick and place / manipulation under the assumption of unsafe factors                    | llama2      |

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
cd imagine_bench
pip install -r requirements.txt
```
Then install the ImagineBench package:
```
pip install -e .
```
If you want to use libero env, please install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) first.

## Basic usage

**Environment** 
```python
import imagine_bench

# Optional task_level: ['real', 'rephrase', 'easy', 'hard'].
env = imagine_bench.make('MetaWorld-v0', level='rephrase')
real_data, imaginary_rollout_rephrase = env.get_dataset(level="rephrase") 

# Or you can use the dataset with other task levels.
env = imagine_bench.make('MetaWorld-v0', level='easy')
real_data, imaginary_rollout_easy = env.get_dataset(level="easy")
```

**Offline RL Training with d3rlpy** 
```python
import d3rlpy
import imagine_bench
from imagine_bench.utils import LlataEncoderFactory, make_d3rlpy_dataset

env = imagine_bench.make('Mujoco-v0', level='rephrase')
real_data, imaginary_rollout_rephrase = env.get_dataset(level="rephrase") 
dataset = make_d3rlpy_dataset(real_data, imaginary_rollout_rephrase)

agent = d3rlpy.algos.TD3PlusBCConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device="cuda:0")

agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name="mujoco",
    )
```