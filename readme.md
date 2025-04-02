# RIMRO: Benchmark for <u>R</u>einforcement Learning from <u>Im</u>aginary <u>Ro</u>llouts

[![Project Status: Early Development](https://img.shields.io/badge/status-early%20development-orange)](https://github.com/yourusername/offline-rl-imaginary-rollouts)

## Overview

A benchmark for evaluating reinforcement learning algorithms that train the policies using both **real data** and **imaginary rollouts from LLMs**. The concept of imaginary rollouts was proposed by [KALM](https://openreview.net/forum?id=tb1MlJCY5g) in NeurIPS 2024, which focuses on extracting knowledge from LLMs, in the form of environmental rollouts, to improve RL policies' performance on novel tasks. Please check the paper for more details.

**Core focus**: Measuring how well agents can learn effective policies through LLM's imaginary rollouts and generalize well on novel tasks.

## 📦 Dataset Status

**Now Available!**  

We have released initial datasets for 2 environments: [CLEVR-Robot](https://github.com/google-research/clevr_robot_env) and [Meta-world](https://github.com/Farama-Foundation/Metaworld), with both real+LLM-generated rollouts.
More environments and tasks are under active development.


We are actively preparing:

- More environment domains
- real+LLM-generated rollouts
- Varying task difficulty levels
