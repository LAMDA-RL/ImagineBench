# Off-policy deep reinforcement learning without exploration
# https://arxiv.org/abs/1812.02900
# https://github.com/sfujim/BCQ

import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.functional import F
from torch.distributions import Normal, kl_divergence
import torch.nn as nn
from scipy.spatial import KDTree
import json
import copy
from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import BasePolicy
from offlinerl.utils.exp import setup_seed
from offlinerl.utils.net.common import BasePolicy
from datetime import datetime
from sklearn.decomposition import PCA
class Actor(nn.Module,BasePolicy):
    def __init__(self, state_dim, action_dim, max_action, device):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.device = device

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    
    def policy_infer(self, obs):
        return self(obs)
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class PRDCPolicy(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        pass


    def forward(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state)
    
    @torch.no_grad()
    def get_action(self, state : np.ndarray):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().detach().numpy().flatten()


def algo_init(args):
    logger.info('Run algo_init function')
    setup_seed(args['seed'])
    if args["state_dim"] and args["action_dim"]:
        state_dim, action_dim = args["state_dim"], args["action_dim"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape
        state_dim, action_dim = get_env_shape(args['task'])
        args["state_dim"], args["action_dim"] = state_dim, action_dim
    else:
        raise NotImplementedError
    device = args['device']
    state_dim = args['state_dim']
    action_dim = args['action_dim']
    max_action = args['max_action']
    discount = args['discount']
    tau = args['tau']
    policy_noise = args['policy_noise']
    noise_clip = args['noise_clip']
    alpha = args['alpha'] 
    policy_freq = args['policy_freq']
    beta = args['beta']
    k = args['k']
    actor_lr = args['actor_lr']
    critic_lr = args['critic_lr'] 

    device = torch.device(device)
    actor = Actor(state_dim, action_dim, max_action, device).to(device)
    actor_target = copy.deepcopy(actor)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    action_dim = action_dim
    max_action = max_action
    discount = discount
    tau = tau
    policy_noise = policy_noise
    noise_clip = noise_clip
    policy_freq = policy_freq
    alpha = alpha

    algo_init = dict(args) 
    algo_init['critic'] = {}
    algo_init['actor'] = {}


    algo_init['critic']['net'] = [critic, critic_target]
    algo_init['critic']['opt'] = critic_optimizer
    algo_init['actor']['net'] = [actor, actor_target]
    algo_init['actor']['opt'] = actor_optimizer 
    return algo_init


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args
        self.actor, self.actor_target = algo_init['actor']['net'] 
        self.critic, self.critic_target = algo_init['critic']['net'] 
        self.critic_optimizer = algo_init['critic']['opt']
        self.actor_optimizer = algo_init['actor']['opt'] 
        for k, v in args.items():
            if k == "actor" or k == "critic":
                continue
            setattr(self, k, v)
        

    def train(self, train_buffer, val_buffer, callback_fn):
        size = train_buffer['obs'].shape[0]
        data = train_buffer.sample(size)
        _s = data['obs'].cpu().numpy()
        _a = data['act'].cpu().numpy()
        data = np.hstack([self.beta * _s, _a])
        self.data = data
        if data.shape[1] > 792:
            self.data_reduced = data[:, 0:792]
            kd_tree = KDTree(self.data_reduced)
        else:
            kd_tree = KDTree(self.data)
        for epoch in range(self.args['max_epoch']):
            for i in range(self.args['steps_per_epoch']):
                batch_data = train_buffer.sample(self.batch_size)
                # batch_data.to_torch(device=self.device)
                obs = batch_data['obs']
                action = batch_data['act']
                next_obs = batch_data['obs_next']
                reward = batch_data['rew']
                done = batch_data['done'].float()

                # train critic
                with torch.no_grad():
                    noise = (torch.randn_like(action) * self.policy_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    )

                    next_action = (self.actor_target(next_obs) + noise).clamp(
                        -self.max_action, self.max_action
                    )

                    # Compute the target Q value
                    target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + (1. - done) * self.discount * target_Q


                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(obs, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                    current_Q2, target_Q
                )
                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                pi = self.actor(obs)
                Q = self.critic.Q1(obs, pi)
                lmbda = self.alpha / Q.abs().mean().detach()
                actor_loss = -lmbda * Q.mean()

                ## Get the nearest neighbor
                key = torch.cat([self.beta * obs, pi], dim=1).detach().cpu().numpy()
                if key.shape[1] > 792:
                    key_reduced = key[:,0:792]  # 降维！
                else:
                    key_reduced = key
                # print(f"{datetime.now().strftime('%Y-%m-%d at %H:%M:%S.%f')[:-3]} | INFO |", i, "line206")
                _, idx = kd_tree.query(key_reduced, k=[self.k], workers=-1)
                # print(f"{datetime.now().strftime('%Y-%m-%d at %H:%M:%S.%f')[:-3]} | INFO |", i, "line208")
                ## Calculate the regularization
                nearest_neightbour = (
                    torch.tensor(self.data[idx][:, :, -self.action_dim:], dtype=torch.float32)
                    .squeeze(dim=1)
                    .to(self.device)
                )
                dc_loss = F.mse_loss(pi, nearest_neightbour)

                # Optimize the actor
                combined_loss = actor_loss + dc_loss
                self.actor_optimizer.zero_grad()
                combined_loss.backward()
                self.actor_optimizer.step()

                # soft target update
                self._sync_weight(self.actor_target, self.actor, soft_target_tau=self.args['tau'])
                self._sync_weight(self.critic_target, self.critic, soft_target_tau=self.args['tau'])

            res = callback_fn(self.get_policy())
            
            res.update({
                "actor_loss" : actor_loss.item(),
                "dc_loss" : dc_loss.item(),
                
            })
            
            self.log_res(epoch, res)

        return self.report_result
    
    def get_policy(self):
        return self.actor