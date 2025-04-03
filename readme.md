<h1>RIMRO: Benchmark for <u>R</u>einforcement Learning from <u>Im</u>aginary <u>Ro</u>llouts</h1>

[![Project Status: Active](https://img.shields.io/badge/status-active-green)](https://github.com/LAMDA-RL/RIMRO)

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

## Basic usage

Pseudo code for using the benchmark:

```python
import rimro

env = rimro.make('MetaWorld-v0')

# Obtain the dataset. Optional task_level: ['rephrase', 'easy', 'hard'].
real_data, imaginary_rollout_rephrasing = env.get_dataset(level="rephrase") 

# Train policy with any offline RL algorithms
policy = offline_rl(real_data, imaginary_rollout_rephrasing)

# Evaluate the policy
eval_result = eval_policy(policy, env, level="rephrase")
```
