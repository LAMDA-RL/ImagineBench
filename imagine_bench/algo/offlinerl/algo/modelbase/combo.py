# COMBO: Conservative Offline Model-Based Policy Optimization
# http://arxiv.org/abs/2102.08363
# No available code

import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Union, Tuple
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.ensemble import EnsembleTransition

from offlinerl.algo.modelbase.model_base import algo_init, ModelBasedAlgoTrainer


class AlgoTrainer(ModelBasedAlgoTrainer):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(algo_init, args)
        
        self.fake_buffer_size = self.args["model_retain_epochs"] * self.args["rollout_batch_size"] * self.args["horizon"]
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim, args['max_epoch'])

        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=self.args['cql_alpha_lr'])
        
    def rollout(self, init_obss, horizon):
        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(horizon):
            if self.args['normalize_obs'] and self.args['policy_scaler']:
                pass
            elif self.args['normalize_obs'] and not self.args['policy_scaler']:
                observations = self.policy_scaler.inverse_transform(observations)
            elif not self.args['normalize_obs'] and self.args['policy_scaler']:
                observations = self.policy_scaler.transform(observations)
            else:
                pass

            if self.args['uniform_rollout']:
                actions = np.random.uniform(
                    self.action_space.low[0],
                    self.action_space.high[0],
                    size=(len(observations), self.action_space.shape[0])
                )
            else:
                actions = self.select_action(observations)

            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions, 
                                                                             transition_scaler=self.args['transition_scaler'],
                                                                             transition_clip=self.args['trainsition_clip'],)

            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean().item()}
    
    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi)

        q1 = self.critic1(obs_to_pred, act)
        q2 = self.critic2(obs_to_pred, act)

        return q1 - log_prob.detach(), q2 - log_prob.detach()
    
    def calc_random_values(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.critic1(obs, random_act)
        q2 = self.critic2(obs, random_act)

        log_prob1 = np.log(0.5**random_act.shape[-1])
        log_prob2 = np.log(0.5**random_act.shape[-1])

        return q1 - log_prob1, q2 - log_prob2
    
    def policy_learn(self, batch: Dict):
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
            mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        batch_size = obss.shape[0]
        
        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update alpha
        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        # compute td error
        if self.args['max_q_backup']:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1).repeat(1, self.args['num_repeat_actions'], 1) \
                                    .view(batch_size * self.args['num_repeat_actions'], next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.target_critic1(tmp_next_obss, tmp_next_actions) \
                                .view(batch_size, self.args['num_repeat_actions'], 1) \
                                .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.target_critic2(tmp_next_obss, tmp_next_actions) \
                                .view(batch_size, self.args['num_repeat_actions'], 1) \
                                .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                next_q = torch.min(
                    self.target_critic1(next_obss, next_actions),
                    self.target_critic2(next_obss, next_actions)
                )
                if not self.args['deterministic_backup']:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        if self.args['rho_s'] == "model":
            obss, actions, next_obss = fake_batch["observations"], fake_batch["actions"], fake_batch["next_observations"]

        batch_size = len(obss)
        random_actions = torch.FloatTensor(batch_size * self.args['num_repeat_actions'], actions.shape[-1]) \
                            .uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1).repeat(1, self.args['num_repeat_actions'], 1) \
                        .view(batch_size * self.args['num_repeat_actions'], obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1).repeat(1, self.args['num_repeat_actions'], 1) \
                            .view(batch_size * self.args['num_repeat_actions'], obss.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        for value in [obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
                      random_value1, random_value2]:
            value.reshape(batch_size, self.args['num_repeat_actions'], 1)

        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)

        # Samples from the original dataset
        real_obss, real_actions = real_batch['observations'], real_batch['actions']
        q1, q2 = self.critic1(real_obss, real_actions), self.critic2(real_obss, real_actions)

        conservative_loss1 = torch.logsumexp(cat_q1 / self.args['temperatue'], dim=1).mean() * self.args['cql_weight'] * self.args['temperatue'] - \
                                q1.mean() * self.args['cql_weight']
        conservative_loss2 = torch.logsumexp(cat_q2 / self.args['temperatue'], dim=1).mean() * self.args['cql_weight'] * self.args['temperatue'] - \
                                q2.mean() * self.args['cql_weight']
        
        if self.args['with_lagrange']:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self.args['lagrange_threshold'])
            conservative_loss2 = cql_alpha * (conservative_loss2 - self.args['lagrange_threshold'])

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()

        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        if self.args['with_lagrange']:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result
        