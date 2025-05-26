import gym
import ray
from copy import deepcopy
import numpy as np
from collections import OrderedDict

from offlinerl.utils.env import get_env
from multiprocessing import Pool


#@ray.remote(num_gpus=0.1)
def test_one_trail(env, policy):
    # env = deepcopy(env)
    # policy = deepcopy(policy)

    state, done = env.reset(), False
    if isinstance(state, tuple):
        state = state[0]
    rewards = 0
    lengths = 0
    while not done:
        state = state[np.newaxis]
        action = policy.get_action(state).reshape(-1)
        result = env.step(action)
        if len(result) == 4:
            state, reward, done, _ = result
        else:
            state, reward, done, timeout,_ = result
            done = done or timeout
        rewards += reward
        lengths += 1

    return (rewards, lengths)

def test_one_trail_sp_local(env, policy):
    # env = deepcopy(env)
    # policy = deepcopy(policy)

    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    while not done:
        state = state.reshape(-1, obs_dim)
        action = policy.get_action(state).reshape(-1, act_dim)
        # print("actions: ", action[0:3,])
        state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1

    return (rewards, lengths)

def test_on_real_env(policy, env, number_of_runs=100):
    rewards = []
    episode_lengths = []
    policy = deepcopy(policy)
    policy.eval()
    
    if (not hasattr(env.spec, "id")) and ("sp" in env._name or "sales" in env._name):
        results = [test_one_trail_sp_local(env, policy) for _ in range(number_of_runs)]
    else:
        pool = Pool(processes=10)
        results = [pool.apply_async(test_one_trail, args=(env, policy)) for _ in range(number_of_runs)]
        results = [result.get() for result in results]
        pool.close()
        pool.join()
    
    policy.train()
    
    rewards = [result[0] for result in results]
    episode_lengths = [result[1] for result in results]
    
    rew_mean = np.mean(rewards)
    rew_std = np.std(rewards)
    len_mean = np.mean(episode_lengths)


    res = OrderedDict()
    res["Reward_Mean_Env"] = rew_mean
    res["Reward_Std_Env"] = rew_std
    res["Length_Mean_Env"] = len_mean
    res["Length_Std_Env"] = np.std(episode_lengths)

    return res
