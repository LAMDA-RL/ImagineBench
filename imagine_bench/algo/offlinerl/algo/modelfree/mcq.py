# Off-policy deep reinforcement learning without exploration
# https://arxiv.org/abs/1812.02900
# https://github.com/sfujim/BCQ

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.functional import F
from torch.distributions import Normal, kl_divergence
from typing import Dict, List, Union, Tuple, Optional

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import MLP,Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_action: Union[int, float],
        device: str = "cpu"
    ) -> None:
        super(VAE, self).__init__()
        self.e1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, output_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        self.to(device=self.device)


    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.relu(self.e1(torch.cat([obs, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)

        return u, mean, std

    def decode(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
        max_action = args["max_action"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape, get_env_action_range
        obs_shape, action_shape = get_env_shape(args['task'])
        max_action, _ = get_env_action_range(args["task"])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
    
    vae = VAE(input_dim=obs_shape, 
              output_dim=action_shape, 
              hidden_dim=750,
              latent_dim=action_shape*2, 
              max_action=max_action,
              device = args['device'])
    vae_optim = torch.optim.Adam(vae.parameters(), lr=args['vae_lr'])
    
    net_a = Net(layer_num = args['actor_layers'], 
                state_shape = obs_shape, 
                hidden_layer_size = args['actor_features'])
    
    actor = TanhGaussianPolicy(preprocess_net = net_a,
                                action_shape = action_shape,
                                hidden_layer_size = args['actor_features'],
                                conditioned_sigma = True,
                              ).to(args['device'])
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    q1 = MLP(obs_shape + action_shape, 1, args['value_features'], args['value_layers'], hidden_activation='relu').to(args['device'])
    q2 = MLP(obs_shape + action_shape, 1, args['value_features'], args['value_layers'], hidden_activation='relu').to(args['device'])
    critic_optim_1 = torch.optim.Adam([*q1.parameters()], lr=args['critic_lr'])
    critic_optim_2 = torch.optim.Adam([*q2.parameters()], lr=args['critic_lr'])

    
    nets =  {
        "vae" : {"net" : vae, "opt" : vae_optim},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "critic" : {"net" : [q1, q2], "opt" : [critic_optim_1,critic_optim_2]},
        
    }
    
    if args["auto_alpha"]:
        if args["target_entropy"]:
            target_entropy = args["target_entropy"]
        else:
            target_entropy = -np.prod(args["action_shape"]).item() 
        log_alpha = torch.zeros(1,requires_grad=True, device=args['device'])
        alpha_optimizer = torch.optim.Adam(
            [log_alpha],
            lr=args["alpha_lr"],
        )
        nets.update({"log_alpha" : {"net" : log_alpha, "opt" : alpha_optimizer, "target_entropy": target_entropy}})
    
    return nets


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.vae = algo_init['vae']['net']
        self.vae_optim = algo_init['vae']['opt']

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim_1 = algo_init['critic']['opt'][0]
        self.critic_optim_2 = algo_init['critic']['opt'][1]

        
        if args["auto_alpha"]:
            self.log_alpha = algo_init["log_alpha"]["net"]
            self.alpha_opt = algo_init["log_alpha"]["opt"]
            self.target_entropy = algo_init["log_alpha"]["target_entropy"]

        self.batch_size = self.args['batch_size']
        self.gamma = self.args['gamma']
        self.lam = self.args['lam']
        self.device = self.args['device']
        self.num_sampled_actions = self.args['num_sampled_actions']
        
    def forward(self, obs, reparameterize=True, return_log_prob=True):
        log_prob = None
        tanh_normal = self.actor(obs,reparameterize=reparameterize,)

        if return_log_prob:
            if reparameterize is True:
                action, pre_tanh_value = tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
            log_prob = tanh_normal.log_prob(
                action,
                pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            if reparameterize is True:
                action = tanh_normal.rsample()
            else:
                action = tanh_normal.sample()
        return action, log_prob
        
    def train(self, train_buffer, val_buffer, callback_fn):
        for epoch in range(self.args['max_epoch']):
            for i in range(self.args['steps_per_epoch']):
                batch_data = train_buffer.sample(self.batch_size)
                batch_data.to_torch(device=self.device)
                
                obs = batch_data['obs']
                action = batch_data['act']
                next_obs = batch_data['obs_next']
                reward = batch_data['rew']
                done = batch_data['done'].float()
                
                new_obs_actions, log_pi = self.forward(obs)
                
                if self.args["auto_alpha"]:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                    self.alpha_opt.zero_grad()
                    alpha_loss.backward()
                    self.alpha_opt.step()
                    alpha = self.log_alpha.exp()
                    alpha = torch.clamp(alpha.detach(), 0.0, 1.0)
                else:
                    alpha_loss = 0
                    alpha = self.args["alpha"]

                # train vae
                recon, mean, std = self.vae(obs, action)
                recon_loss = F.mse_loss(recon, action)
                KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                vae_loss = recon_loss + KL_loss
                
                self.vae_optim.zero_grad()
                vae_loss.backward()
                self.vae_optim.step()

                # train critic
                with torch.no_grad():
                    next_action, next_log_prob = self.forward(next_obs)
                    next_obs_action = torch.cat([next_obs, next_action], dim=-1)
                    next_q = torch.min(
                        self.target_q1(next_obs_action), self.target_q2(next_obs_action)
                    ) - alpha * next_log_prob
                    target_q_for_in_actions = reward + self.gamma * (1 - done) * next_q
                obs_action = torch.cat([obs, action], dim=-1)
                q1_in, q2_in = self.q1(obs_action), self.q2(obs_action)
                critic1_loss_for_in_actions = ((q1_in - target_q_for_in_actions).pow(2)).mean()
                critic2_loss_for_in_actions = ((q2_in - target_q_for_in_actions).pow(2)).mean()
                
                s_in = torch.cat([obs, next_obs], dim=0)
                with torch.no_grad():
                    s_in_repeat = torch.repeat_interleave(s_in, self.num_sampled_actions, 0)
   
                    sampled_actions = self.vae.decode(s_in_repeat)
                    repeat_obs_action = torch.cat([s_in_repeat, sampled_actions], dim=-1)
                    target_q1_for_ood_actions = self.target_q1(repeat_obs_action).reshape(s_in.shape[0], -1).max(1)[0].reshape(-1, 1)
                    target_q2_for_ood_actions = self.target_q2(repeat_obs_action).reshape(s_in.shape[0], -1).max(1)[0].reshape(-1, 1)
                    target_q_for_ood_actions = torch.min(target_q1_for_ood_actions, target_q2_for_ood_actions)
                    ood_actions, _ = self.forward(s_in)
                
                ood_obs_action = torch.cat([s_in, ood_actions], dim=-1)
                q1_ood, q2_ood = self.q1(ood_obs_action), self.q2(ood_obs_action)
                critic1_loss_for_ood_actions = ((q1_ood - target_q_for_ood_actions).pow(2)).mean()
                critic2_loss_for_ood_actions = ((q2_ood - target_q_for_ood_actions).pow(2)).mean()

                critic1_loss = self.lam * critic1_loss_for_in_actions + (1 - self.lam) * critic1_loss_for_ood_actions
                self.critic_optim_1.zero_grad()
                critic1_loss.backward()
                self.critic_optim_1.step()

                critic2_loss = self.lam * critic2_loss_for_in_actions + (1 - self.lam) * critic2_loss_for_ood_actions
                self.critic_optim_2.zero_grad()
                critic2_loss.backward()
                self.critic_optim_2.step()

                # update actor
                _action, log_probs = self.forward(obs)
                obs_action = torch.cat([obs, _action], dim=-1)
                q1a, q2a = self.q1(obs_action), self.q2(obs_action)
                
                actor_loss = - torch.min(q1a, q2a).mean() + alpha * log_probs.mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # soft target update
                self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
                self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])
                

            res = callback_fn(self.get_policy())

            if self.args["auto_alpha"]:
                alpha = alpha.item()
                alpha_loss = alpha_loss.item()
                
            res.update({
                "actor_loss" : actor_loss.item(),
                "critic1_loss" : critic1_loss.item(),
                "critic2_loss" : critic2_loss.item(),
                "alpha_loss" : alpha_loss,
                "vae_loss" : vae_loss.item(),
                "alpha" : alpha,
                
            })

            self.log_res(epoch, res)

        return self.report_result

    
    def get_model(self):
        return self.actor
    
    def get_policy(self):
        return self.actor