# RIMRO: Benchmark for <u>R</u>einforcement Learning from <u>Im</u>aginary <u>Ro</u>llouts

[![Project Status: Early Development](https://img.shields.io/badge/status-active-green)](https://github.com/LAMDA-RL/RIMRO)

## Overview

A benchmark for evaluating reinforcement learning algorithms that train the policies using both **real data** and **imaginary rollouts from LLMs**. The concept of imaginary rollouts was proposed by [KALM](https://openreview.net/forum?id=tb1MlJCY5g) (NeurIPS 2024), which focuses on extracting knowledge from LLMs, in the form of environmental rollouts, to improve RL policies' performance on novel tasks. Please check the paper for more details.

**Core focus**: Measuring how well agents can learn effective policies through LLM's imaginary rollouts and generalize well on novel tasks.


## 📢 News
- **Mar 31, 2025**: Initial release of datasets for CLEVR-Robot and Meta-World and the environment with Gymnasium wrapper.


## Dataset Status

**Now Available!**  

We have released initial datasets for 2 environments: [CLEVR-Robot](https://github.com/google-research/clevr_robot_env) and [Meta-world](https://github.com/Farama-Foundation/Metaworld), with both real+LLM-generated rollouts.
More environments and tasks are under active development.

### Available Environments
| Environment | Tasks                                  | LLM Sources |
|-------------|----------------------------------------|-------------|
| CLEVR-Robot | moving 5 balls to target configuration | llama2      |
| Meta-World  | distinct robotic manipulation tasks    | llama2      |
---

We are actively preparing:

- More environment domains
- real+LLM-generated rollouts
- Varying task difficulty levels

## Basic usage

We provide a pseudo code for using the benchmark:

```python
import gym
import envs

env = gym.make('ball')
real_data, imaginary_rollout_rephrasing = env.get_dataset()

# Train policy with any offline RL algorithms
policy = offline_rl(real_data, imaginary_rollout_rephrasing)

# Evaluate the policy
eval_result = eval_policy(policy, env, task_level="rephrasing")
```