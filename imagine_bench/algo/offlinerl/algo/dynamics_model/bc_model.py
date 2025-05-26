import os
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import json

import torch
import numpy as np
from loguru import logger
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
    dynamics_scaler = StandardScaler()

    _termination_fn = get_termination_fn(task=args['task'])

    return {
        "task_info" : {"obs_shape" : obs_shape, "action_shape" : action_shape, "action_dim": action_dim,
                       "obs_space": obs_space, "action_space": action_space},
        "transition" : {"net" : dynamics_model, "opt" : dynamics_optim,# "adv_opt" : dynamics_adv_optim, 
                        "scaler": dynamics_scaler, "_terminal_fn" : _termination_fn},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.obs_shape = algo_init['task_info']['obs_shape']
        self.action_shape = algo_init['task_info']['action_shape']
        self.action_dim = algo_init['task_info']['action_dim']
        self.obs_space = algo_init['task_info']['obs_space']
        self.action_space = algo_init['task_info']['action_space']

        self.dynamics_model = algo_init['transition']['net']
        self.dynamics_optim = algo_init['transition']['opt']
        # self.dynamics_adv_optim = algo_init['transition']['adv_opt']
        self.dynamics_scaler = algo_init['transition']['scaler']
        self._terminal_fn = algo_init['transition']['_terminal_fn']

        self.device = args['device']

    def set_data_buffer(self, buffer):
        real_buffer = ReplayBuffer(buffer_size=len(buffer["obs"]),
                                   obs_shape=self.args['obs_shape'],
                                   obs_dtype=np.float32,
                                   action_dim=self.action_dim,
                                   action_dtype=np.float32,
                                   device=self.args['device']
                                   )
        real_buffer.load_dataset(buffer)
        self.obs_max = np.concatenate([real_buffer.observations, real_buffer.next_observations], axis=0).max(axis=0)
        self.obs_min = np.concatenate([real_buffer.observations, real_buffer.next_observations], axis=0).min(axis=0)
        self.rew_max = np.concatenate([real_buffer.rewards, real_buffer.rewards], axis=0).max(axis=0)
        self.rew_min = np.concatenate([real_buffer.rewards, real_buffer.rewards], axis=0).min(axis=0)

        obs_mean, obs_std = real_buffer.normalize_obs(inplace=self.args['normalize_obs'])
        return real_buffer, obs_mean, obs_std

    def set_ensemble_dynamics(self, obs_mean, obs_std):
        termination_fn = obs_unnormalization(self._terminal_fn, obs_mean, obs_std) if self.args["normalize_obs"] else self._terminal_fn
        policy_scaler = StandardScaler(mu=obs_mean, std=obs_std)
        dynamics = EnsembleDynamics(self.dynamics_model,
                                    self.dynamics_optim,
                                    self.dynamics_scaler,
                                    termination_fn,
                                    data_range=(self.obs_min, self.obs_max, self.rew_min, self.rew_max),
                                    )
        return termination_fn, dynamics, policy_scaler
    
    def train(self, train_buffer, val_buffer, callback_fn):
        total_buffer = train_buffer
        if val_buffer is not None:
            for k, v in train_buffer.items():
                total_buffer[k] = np.concatenate([total_buffer[k], val_buffer[k]], axis=0)

        self.real_buffer, obs_mean, obs_std = self.set_data_buffer(total_buffer)
        self.termination_fn, self.dynamics, self.policy_scaler = self.set_ensemble_dynamics(obs_mean, obs_std)

        if self.args['dynamics_path'] is not None:
            if os.path.exists(self.args['dynamics_path']):
                load_path = self.args['dynamics_path']
            else:
                common_dynamcis_dir = os.path.join(self.index_path, "dynamics_model")
                load_path = os.path.join(common_dynamcis_dir, self.args['dynamics_path'])
                print(f"load_path: {load_path}")
            self.dynamics.model = torch.load(os.path.join(load_path, "dynamics_model.pt"), map_location='cpu').to(self.device)
            self.dynamics.model.device = torch.device(self.device)
            self.dynamics.scaler.load_scaler(load_path, surfix="dynamics_")
        else:
            res = self.train_transition(self.real_buffer.sample_all(), callback_fn)
            # if self.args['dynamics_save_path'] is not None:
            # create common directory for dynamics model
            common_dynamcis_dir = os.path.join(self.index_path, "dynamics_model")
            create_dir(common_dynamcis_dir)
            # create specific directory for saving dynamics model
            run_id = self.exp_run.name.split( )[-1]
            dynamic_save_dir = os.path.join(common_dynamcis_dir, run_id)
            if not os.path.exists(dynamic_save_dir):
                os.makedirs(dynamic_save_dir)
            # save dynamics model
            model_save_path = os.path.join(dynamic_save_dir, "dynamics_model.pt")
            torch.save(self.dynamics.model, model_save_path)
            self.dynamics.scaler.save_scaler(dynamic_save_dir, surfix="dynamics_")
            
            res["model_save_path"] = model_save_path
            res["hparams"] = self.exp_run['hparams']
            
        return res
    
    def train_transition(self, data, callback_fn):
        inputs, targets = self.dynamics.format_samples_for_training(data)
        data_size = inputs.shape[0]
        val_size = min(int(data_size * self.args['val_ratio']), 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size), generator=torch.Generator().manual_seed(42))
        print(val_splits.indices[50:60])
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        val_inputs, val_targets = inputs[val_splits.indices], targets[val_splits.indices]

        self.dynamics.scaler.fit(train_inputs)
        if self.args["transition_scaler"]:
            train_inputs = self.dynamics.scaler.transform(train_inputs)
            val_inputs = self.dynamics.scaler.transform(val_inputs)
        val_losses = [1e10 for i in range(self.dynamics.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.dynamics.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]
        
        # train_inputs = torch.as_tensor(train_inputs).to(self.dynamics.model.device)
        # train_targets = torch.as_tensor(train_targets).to(self.dynamics.model.device)
        val_inputs = torch.as_tensor(val_inputs).to(self.dynamics.model.device)
        val_targets = torch.as_tensor(val_targets).to(self.dynamics.model.device)

        epoch = 0
        cnt = 0
        while True:
            epoch += 1
            train_loss = self.dynamics_learn(train_inputs[data_idxes], train_targets[data_idxes], self.args['transition_batch_size'], self.args['logvar_loss_coef'])
            # new_val_losses = self.validate(val_inputs, val_targets)
            new_val_losses = callback_fn(val_inputs, val_targets, self.dynamics.model)
            print(f"epoch: {epoch}, val loss: {new_val_losses}")

            val_loss = (np.sort(new_val_losses)[:self.dynamics.model.num_elites]).mean().item()
            if epoch > 1:
                self.log_res(epoch, {"loss/dynamics_train_loss": train_loss, "loss/dynamics_val_loss": val_loss, "loss/least_metric": np.mean(val_losses).item()})

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 1e-4:
                    indexes.append(i)
                    val_losses[i] = new_loss

            if len(indexes) > 0:
                self.dynamics.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1

            if (cnt >= self.args['max_epochs_since_update']) or (self.args['transition_max_epochs'] is not None and (epoch >= self.args['transition_max_epochs'])):
                break

        indexes = self.dynamics.select_elites(val_losses)
        self.dynamics.model.set_elites(indexes)
        self.dynamics.model.load_save()
        # self.dynamics.save(logger.model_dir)
        self.dynamics.model.eval()
        print("elites:{} , val loss: {}".format(indexes, (np.sort(val_losses)[:self.dynamics.model.num_elites]).mean()))
        print(f"val loss each elite: {np.sort(val_losses)[:self.dynamics.model.num_elites]}")
        
        res = {"loss" : np.mean(val_losses).item()}
        return res

    def dynamics_learn(self, inputs, targets, batch_size, logvar_loss_coef):
        self.dynamics.model.train()
        train_size = inputs.shape[1]
        losses = []

        inputs = torch.as_tensor(inputs).to(self.dynamics.model.device)
        targets = torch.as_tensor(targets).to(self.dynamics.model.device)
        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            if isinstance(targets_batch, np.ndarray):
                targets_batch = torch.as_tensor(targets_batch).to(self.dynamics.model.device)
            
            mean, logvar = self.dynamics.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.dynamics.model.get_decay_loss()
            loss = loss + logvar_loss_coef * \
                    self.dynamics.model.max_logvar.sum() - \
                    logvar_loss_coef * self.dynamics.model.min_logvar.sum()

            self.dynamics_optim.zero_grad()
            loss.backward()
            self.dynamics_optim.step()

            losses.append(loss.item())
        return np.mean(losses).item()
    
    def get_policy(self,):
        pass

    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray):
        self.dynamics.model.eval()
        if isinstance(targets, np.ndarray):
            targets = torch.as_tensor(targets).to(self.dynamics.model.device)
        mean, _ = self.dynamics.model(inputs)
        # if not self.args["normalize_obs"]:
        #     norm_delta_obs = self.policy_scaler.transform(mean[..., :-1])
        #     norm_mean = np.concatenate([norm_delta_obs, mean[..., -1:]], axis=-1)
        #     norm_target_obs = self.policy_scaler.transform(targets[..., :-1])
        #     norm_target = np.concatenate([norm_target_obs, targets[..., -1:]], axis=-1)
        # else:
        #     norm_mean = mean
        #     norm_target = targets
        loss = ((mean - targets) ** 2).mean(axis=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss

  