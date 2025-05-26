# A Minimalist Approach to Offline Reinforcement Learning
# https://arxiv.org/pdf/2106.06860
# https://github.com/sfujim/TD3_BC
import torch
from copy import deepcopy
from loguru import logger
from torch.functional import F

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import MLP,Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed


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
    
    net_a = Net(layer_num = args['actor_layers'], 
                state_shape = obs_shape, 
                hidden_layer_size = args['actor_features'])
    
    actor = TanhGaussianPolicy(preprocess_net = net_a,
                               action_shape = action_shape,
                               hidden_layer_size = args['actor_features'],
                               conditioned_sigma = True,
                              ).to(args['device'])
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    critic_1 = MLP(obs_shape + action_shape, 1, args['value_features'], args['value_layers'], hidden_activation='relu').to(args['device'])
    critic_2 = MLP(obs_shape + action_shape, 1, args['value_features'], args['value_layers'], hidden_activation='relu').to(args['device'])
    critic_1_optim = torch.optim.Adam([*critic_1.parameters()], lr=args['critic_lr'])
    critic_2_optim = torch.optim.Adam([*critic_2.parameters()], lr=args['critic_lr'])
    
    nets =  {
        "actor" : {"net" : actor, "opt" : actor_optim},
        "critic" : {"net" : [critic_1, critic_2], "opt" : [critic_1_optim,critic_2_optim]},
        
    }
    
    return nets


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.critic_1, self.critic_2 = algo_init['critic']['net']
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)
        self.critic_1_optim = algo_init['critic']['opt'][0]
        self.critic_2_optim = algo_init['critic']['opt'][1]

        self.alpha = self.args['alpha']
        self.policy_noise = self.args['policy_noise']
        self.noise_clip = self.args['noise_clip']
        self.policy_freq = self.args['policy_freq']
        self.discount = self.args['discount']
        
        self.batch_size = self.args['batch_size']
        self.device = self.args['device']
        self.max_action = 1
        
        
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
        # train_buffer
        obs_mean = train_buffer["obs"].mean(0)
        obs_std = train_buffer["obs"].std(0) + 1e-3
        obs_mean = torch.as_tensor(obs_mean, dtype=torch.float32)
        obs_std = torch.as_tensor(obs_std, dtype=torch.float32)
        self.actor.preprocess.s_mean = obs_mean
        self.actor.preprocess.s_std = obs_std
        
        self.target_actor = deepcopy(self.actor)
        
        for epoch in range(self.args['max_epoch']):
            for i in range(self.args['steps_per_epoch']):
                batch_data = train_buffer.sample(self.batch_size)
                # batch_data.to_torch(device=self.device)
                
                obs = batch_data['obs']
                action = batch_data['act']
                next_obs = batch_data['obs_next']
                reward = batch_data['rew']
                done = batch_data['done'].float()
                
                with torch.no_grad():
                    noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    next_action = (self.target_actor(next_obs).mode + noise).clamp(-self.max_action, self.max_action)
                    next_obs_action = torch.cat([next_obs, next_action], dim=-1)
                    target_q = torch.min(
                        self.target_critic_1(next_obs_action), self.target_critic_2(next_obs_action)
                    )*self.discount*(1-done) + reward
                
                obs_action = torch.cat([obs, action], dim=-1)
                current_q1, current_q2 = self.critic_1(obs_action), self.critic_2(obs_action)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                
                # Optimize the critic
                self.critic_1_optim.zero_grad()
                self.critic_2_optim.zero_grad()
                critic_loss.backward()
                self.critic_1_optim.step()
                self.critic_2_optim.step()
                
                
                if i % self.policy_freq == 0:
                    pi = self.actor(obs).mode
                    q = self.critic_1(torch.cat([obs, pi], dim=-1))
                    lmbda = self.alpha / q.abs().mean().detach()
                    actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
                    
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    
                    self._sync_weight(self.target_actor, self.actor, soft_target_tau=self.args['soft_target_tau'])
                    self._sync_weight(self.target_critic_1, self.critic_1, soft_target_tau=self.args['soft_target_tau'])
                    self._sync_weight(self.target_critic_2, self.critic_2, soft_target_tau=self.args['soft_target_tau'])
                
            res = callback_fn(self.get_policy())
                
            res.update({
                "actor_loss" : actor_loss.item(),
                "critic_loss" : critic_loss.item(),
                "lmbda" : lmbda.item(),
                "q" : q.mean().item(),
            })


            self.log_res(epoch, res)

        return self.report_result

    def get_model(self):
        return self.actor
    
    def get_policy(self):
        return self.actor