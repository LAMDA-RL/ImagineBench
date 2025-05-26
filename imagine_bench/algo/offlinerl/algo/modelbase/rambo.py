import os
from copy import deepcopy
from collections import defaultdict
import json

import torch
import numpy as np
from loguru import logger
from typing import Dict
from operator import itemgetter

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.exp import setup_seed
from offlinerl.utils.net.terminal_check import get_termination_fn
from offlinerl.utils.env import get_env_obs_act_spaces
from offlinerl.utils.io import create_dir

from offlinerl.outside_utils.nets import MLP
from offlinerl.outside_utils.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerl.outside_utils.dynamics import EnsembleDynamics
from offlinerl.outside_utils.utils.scaler import StandardScaler
from offlinerl.outside_utils.utils.termination_fns import obs_unnormalization
from offlinerl.outside_utils.buffer import ReplayBuffer

from offlinerl.algo.modelbase.model_base import ModelBasedAlgoTrainer


def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape
        obs_shape, action_shape = get_env_shape(args['task'])
    else:
        raise NotImplementedError

    obs_space, action_space = get_env_obs_act_spaces(args['task'])
    obs_shape, action_shape = (obs_shape,), (action_shape,)
    args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    action_dim = np.prod(action_shape)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=args['hidden_dims'])
    critic1_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=args['hidden_dims'])
    critic2_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=args['hidden_dims'])
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args['device'])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    critic1 = Critic(critic1_backbone, args['device'])
    critic2 = Critic(critic2_backbone, args['device'])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args['critic_lr'])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args['critic_lr'])

    if args['learnable_alpha']:
        target_entropy = args['target_entropy'] if args['target_entropy'] is not None else -np.prod(action_shape)

        log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
        alpha_optim = torch.optim.Adam([log_alpha], lr=args['alpha_lr'])
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args['alpha']
        alpha_optim = None

    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(obs_shape),
        action_dim=action_dim,
        hidden_dims=args['dynamics_hidden_dims'],
        num_ensemble=args['transition_init_num'],
        num_elites=args['transition_select_num'],
        weight_decays=args['dynamics_weight_decay'],
        device=args['device']
    )
    dynamics_optim = torch.optim.Adam(dynamics_model.parameters(), lr=args['transition_lr'])
    dynamics_adv_optim = torch.optim.Adam(dynamics_model.parameters(), lr=args['transition_adv_lr'])
    dynamics_scaler = StandardScaler()

    _termination_fn = get_termination_fn(task=args['task'])

    return {
        "task_info" : {"obs_shape" : obs_shape, "action_shape" : action_shape, "action_dim": action_dim,
                        "obs_space": obs_space, "action_space": action_space},
        "transition" : {"net" : dynamics_model, "opt" : dynamics_optim, "adv_opt" : dynamics_adv_optim, 
                        "scaler": dynamics_scaler, "_terminal_fn" : _termination_fn},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "alpha" : {"net" : alpha, "opt" : alpha_optim},
        "critic" : {"net" : [critic1, critic2], "opt" : [critic1_optim, critic2_optim]},
    }


class AlgoTrainer(ModelBasedAlgoTrainer):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(algo_init, args)

        self.dynamics_adv_optim = algo_init['transition']['adv_opt']
        self.fake_buffer_size = self.args['steps_per_epoch'] // self.args["rollout_freq"] * \
                                self.args["model_retain_epochs"] * self.args["rollout_batch_size"] * self.args["horizon"]

    def train(self, train_buffer, val_buffer, callback_fn):
        total_buffer = train_buffer
        if val_buffer is not None:
            for k, v in train_buffer.items():
                total_buffer[k] = np.concatenate([total_buffer[k], val_buffer[k]], axis=0)

        self.real_buffer, self.fake_buffer, obs_mean, obs_std = self.set_data_buffer(total_buffer)
        self.termination_fn, self.dynamics, self.policy_scaler = self.set_ensemble_dynamics(obs_mean, obs_std)
        if self.args["policy_scaler"]:
            self.actor.set_scaler(self.policy_scaler)

        if self.args['policy_bc_path'] is not None:
            self.actor = torch.load(os.path.join(self.args['policy_bc_path'], "actor.pt"), map_location='cpu').to(self.device)
            if self.args["policy_scaler"]:
                self.actor.scaler.load_scaler(self.args['policy_bc_path'], surfix="policy_")
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args['actor_lr'])
        else:
            self.pretrain_policy(self.real_buffer.sample_all())
            if self.args['policy_bc_save_path'] is not None: 
                torch.save(self.actor, os.path.join(self.args['policy_bc_save_path'], "actor.pt"))
                if self.args["policy_scaler"]:
                    self.actor.scaler.save_scaler(self.args['policy_bc_save_path'], surfix="policy_")

        if self.args['dynamics_path'] is not None:
            if os.path.exists(self.args['dynamics_path']):
                load_path = self.args['dynamics_path']
            else:
                common_dynamcis_dir = os.path.join(self.index_path, "dynamics_model")
                load_path = os.path.join(common_dynamcis_dir, self.args['dynamics_path'])
                print(f"load_path: {load_path}")
            if load_path.endswith("dynamics_model.pt"):
                load_path = os.path.dirname(load_path)
            self.dynamics.model = torch.load(os.path.join(load_path, "dynamics_model.pt"), map_location='cpu').to(self.device)
            self.dynamics.model.device = torch.device(self.device)
            self.dynamics.scaler.load_scaler(load_path, surfix="dynamics_")
            self.dynamics_adv_optim = torch.optim.Adam(self.dynamics.model.parameters(), lr=self.args['transition_adv_lr'])
        else:
            self.train_transition(self.real_buffer.sample_all())
            # if self.args['dynamics_save_path'] is not None: 
            #     torch.save(self.dynamics.model, os.path.join(self.args['dynamics_save_path'], "dynamics_model.pt"))
            #     self.dynamics.scaler.save_scaler(self.args['dynamics_save_path'], surfix="dynamics_")
            # create common directory for dynamics model
            common_dynamcis_dir = os.path.join(self.index_path, "dynamics_model")
            create_dir(common_dynamcis_dir)
            # create specific directory for saving dynamics model
            run_id = self.exp_run.name.split( )[-1]
            dynamic_save_dir = os.path.join(common_dynamcis_dir, run_id)
            if not os.path.exists(dynamic_save_dir):
                os.makedirs(dynamic_save_dir)
            # save dynamics model
            torch.save(self.dynamics.model, os.path.join(dynamic_save_dir, "dynamics_model.pt"))
            self.dynamics.scaler.save_scaler(dynamic_save_dir, surfix="dynamics_")

        self.train_policy(callback_fn)
        
        return self.report_result

    def pretrain_policy(self, data) -> None:
        batch_size = self.args['policy_bc_batch_size']
        self._bc_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args['policy_bc_lr'])
        observations = data["observations"]
        if self.args["normalize_obs"] and self.args['policy_scaler']:
            pass
        elif self.args["normalize_obs"] and not self.args['policy_scaler']:
            observations = self.policy_scaler.inverse_transform(observations)
        elif not self.args["normalize_obs"] and self.args['policy_scaler']:
            observations = self.policy_scaler.transform(observations)
        else:
            pass
        actions = data["actions"]
        sample_num = observations.shape[0]
        idxs = np.arange(sample_num)
        print("Pretraining policy")
        self.actor.train()
        for epoch in range(self.args['policy_bc_epoch']):
            np.random.shuffle(idxs)
            sum_loss = 0
            for i_batch in range(sample_num // batch_size):
                batch_obs = observations[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_act = actions[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_act = torch.from_numpy(batch_act).to(self.device)
                dist = self.actor(batch_obs)
                pred_actions, _ = dist.rsample()
                bc_loss = ((pred_actions - batch_act) ** 2).mean()

                self._bc_optim.zero_grad()
                bc_loss.backward()
                self._bc_optim.step()
                sum_loss += bc_loss.cpu().item()
            print(f"Epoch {epoch}, mean bc loss {sum_loss/i_batch}")
        return  
    
    def policy_learn(self, batch: Dict):
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
            mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.target_critic1(next_obss, next_actions), self.target_critic2(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result
    
    def update_transition(self, real_buffer):
        all_loss_info = {"adv_dynamics_update/all_loss": 0, 
                         "adv_dynamics_update/sl_loss": 0, 
                         "adv_dynamics_update/adv_loss": 0, 
                         "adv_dynamics_update/adv_advantage": 0, 
                         "adv_dynamics_update/adv_log_prob": 0, 
                         }
        self.dynamics.model.train()
        steps = 0
        while steps < self.args['adv_train_steps']:
            init_obss = real_buffer.sample(self.args['adv_rollout_batch_size'])["observations"].cpu().numpy()
            observations = init_obss
            for t in range(self.args['adv_rollout_length']):
                actions = self.select_action(observations)
                sl_observations, sl_actions, sl_next_observations, sl_rewards = \
                    itemgetter("observations", "actions", "next_observations", "rewards")(real_buffer.sample(self.args['adv_rollout_batch_size']))
                next_observations, terminals, loss_info = self.dynamics_step_and_forward(observations, actions, sl_observations, sl_actions, sl_next_observations, sl_rewards)
                if self.args['trainsition_clip']:
                    next_observations = np.clip(next_observations, self.obs_min, self.obs_max)
                for _key in loss_info:
                    all_loss_info[_key] += loss_info[_key]
                steps += 1
                observations = next_observations.copy()
                if steps == 1000:
                    break
        self.dynamics.model.eval()
        return {_key: _value / steps for _key, _value in all_loss_info.items()}

    def dynamics_step_and_forward(self, observations, actions, sl_observations, sl_actions, sl_next_observations, sl_rewards):
        obs_act = np.concatenate([observations, actions], axis=-1)
        if self.args["transition_scaler"]:
            obs_act = self.dynamics.scaler.transform(obs_act)
        diff_mean, logvar = self.dynamics.model(obs_act)
        observations = torch.from_numpy(observations).to(diff_mean.device)
        diff_obs, diff_reward = torch.split(diff_mean, [diff_mean.shape[-1]-1, 1], dim=-1)
        mean = torch.cat([diff_obs + observations, diff_reward], dim=-1)
        std = torch.sqrt(torch.exp(logvar))
        dist = torch.distributions.Normal(mean, std)
        ensemble_sample = dist.sample()
        ensemble_size, batch_size, _ = ensemble_sample.shape

        # select the next observations
        selected_indexes = self.dynamics.model.random_elite_idxs(batch_size)
        sample = ensemble_sample[selected_indexes, np.arange(batch_size)]
        next_observations = sample[..., :-1]
        rewards = sample[..., -1:]
        terminals = self.dynamics.terminal_fn(observations.detach().cpu().numpy(), actions, next_observations.detach().cpu().numpy())

        # compute logprob
        log_prob = dist.log_prob(sample).sum(-1, keepdim=True)
        log_prob = log_prob[self.dynamics.model.elites.data, ...]
        prob = log_prob.double().exp()
        prob = prob * (1/len(self.dynamics.model.elites.data))
        log_prob = prob.sum(0).log().type(torch.float32)

        # compute the advantage
        with torch.no_grad():
            next_actions, next_policy_log_prob = self.actforward(next_observations, deterministic=True)
            next_q = torch.minimum(
                self.critic1(next_observations, next_actions), 
                self.critic2(next_observations, next_actions)
            )
            if self.args['include_ent_in_adv']:
                next_q = next_q - self._alpha * next_policy_log_prob

            value = rewards + (1-torch.from_numpy(terminals).to(mean.device).float()) * self._gamma * next_q

            value_baseline = torch.minimum(
                self.critic1(observations, actions), 
                self.critic2(observations, actions)
            )
            advantage = value - value_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std()+1e-6)
        adv_loss = (log_prob * advantage).mean()

        # compute the supervised loss
        sl_input = torch.cat([sl_observations, sl_actions], dim=-1).cpu().numpy()
        sl_target = torch.cat([sl_next_observations - sl_observations, sl_rewards], dim=-1)
        if self.args["transition_scaler"]:
            sl_input = self.dynamics.scaler.transform(sl_input)
        sl_mean, sl_logvar = self.dynamics.model(sl_input)
        sl_inv_var = torch.exp(-sl_logvar)
        sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2))
        sl_var_loss = sl_logvar.mean(dim=(1, 2))
        sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
        sl_loss = sl_loss + self.dynamics.model.get_decay_loss()
        sl_loss = sl_loss + 0.001 * self.dynamics.model.max_logvar.sum() - 0.001 * self.dynamics.model.min_logvar.sum()

        all_loss = self.args['adv_weight'] * adv_loss + sl_loss

        self.dynamics_adv_optim.zero_grad()
        all_loss.backward()
        self.dynamics_adv_optim.step()

        return next_observations.cpu().numpy(), terminals, {"adv_dynamics_update/adv_loss": adv_loss.cpu().item(), 
                                                            "adv_dynamics_update/sl_loss": sl_loss.cpu().item(),
                                                            "adv_dynamics_update/all_loss": all_loss.cpu().item(),
                                                            "adv_dynamics_update/adv_advantage": advantage.mean().cpu().item(),
                                                            "adv_dynamics_update/adv_log_prob": log_prob.mean().cpu().item()}
