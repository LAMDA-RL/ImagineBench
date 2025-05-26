import torch
import numpy as np
from copy import deepcopy
from loguru import logger
import ray
import os

from offlinerl.utils.env import get_env
from offlinerl.algo.base import BaseAlgo
from collections import OrderedDict
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed
import offlinerl.utils.loader as loader
from offlinerl.utils.net.terminal_check import get_termination_fn
from offlinerl.utils.env import get_env_obs_act_spaces
from offlinerl.utils.io import create_dir

from offlinerl.utils.data import ModelBuffer
# from offlinerl.utils.net.model.ensemble import EnsembleTransition
from offlinerl.utils.net.model_GRU import GRU_Model
from offlinerl.utils.net.maple_actor import Maple_actor
from offlinerl.utils.net.model.maple_critic import Maple_critic
from offlinerl.utils.simple_replay_pool import SimpleReplayTrajPool
from offlinerl.utils import loader

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
    obs_dim = np.prod(obs_shape)
    action_dim = np.prod(action_shape)

    if 'data_name' not in args:
        args['data_name'] = args['task'][5:]

    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args['dynamics_hidden_dims'],
        num_ensemble=args['transition_init_num'],
        num_elites=args['transition_select_num'],
        weight_decays=args['dynamics_weight_decay'],
        device=args['device']
    )
    dynamics_optim = torch.optim.Adam(dynamics_model.parameters(), lr=args['transition_lr'])
    dynamics_scaler = StandardScaler()

    policy_gru = GRU_Model(obs_dim, action_dim, args['device'],args['lstm_hidden_unit']).to(args['device'])
    value_gru = GRU_Model(obs_dim, action_dim, args['device'], args['lstm_hidden_unit']).to(args['device'])
    actor = Maple_actor(obs_dim, action_dim).to(args['device'])
    q1 = Maple_critic(obs_dim, action_dim).to(args['device'])
    q2 = Maple_critic(obs_dim, action_dim).to(args['device'])

    actor_optim = torch.optim.Adam([*policy_gru.parameters(),*actor.parameters()], lr=args['actor_lr'])

    # set alpha
    if args['learnable_alpha']:
        target_entropy = args['target_entropy'] if args['target_entropy'] is not None else -np.prod(action_shape)

        log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
        alpha_optim = torch.optim.Adam([log_alpha], lr=args['alpha_lr'])
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args['alpha']
        alpha_optim = None

    critic_optim = torch.optim.Adam([*value_gru.parameters(),*q1.parameters(), *q2.parameters()], lr=args['critic_lr'])
    _termination_fn = get_termination_fn(task=args['task'])

    return {
        "task_info" : {"obs_shape" : obs_shape, "action_shape" : action_shape, "action_dim": action_dim,
                       "obs_space": obs_space, "action_space": action_space},
        "transition": {"net": dynamics_model, "opt": dynamics_optim,
                       "scaler": dynamics_scaler, "_terminal_fn" : _termination_fn},
        "actor": {"net": [policy_gru,actor], "opt": actor_optim},
        "alpha" : {"net" : alpha, "opt" : alpha_optim},
        "critic": {"net": [value_gru, q1, q2], "opt": critic_optim},
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
        self.dynamics_scaler = algo_init['transition']['scaler']
        self._terminal_fn = algo_init['transition']['_terminal_fn']

        self.policy_gru, self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self._is_auto_alpha = False
        self.alpha_optim = algo_init['alpha']['opt']
        if self.alpha_optim is not None:
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = algo_init['alpha']['net']
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = algo_init['alpha']['net']

        self.value_gru, self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init['critic']['opt']
        
        self.device = args['device']

        self.args['buffer_size'] = int(self.args['data_collection_per_epoch']) * self.args['horizon'] * 5
        self.args['model_pool_size'] = int(args['model_pool_size'])

        self.is_terminal = lambda obs, act, next_obs: self._terminal_fn(obs, act, next_obs).astype(bool)

        # self.fake_buffer_size = self.args["model_retain_epochs"] * self.args["rollout_batch_size"] * self.args["horizon"]
        self.lr_scheduler = None

        # for mopo
        self.penalty_coef = 0.0 if "penalty_coef" not in self.args else self.args["penalty_coef"]
        self.uncertainty_mode = "None" if "uncertainty_mode" not in self.args else self.args["uncertainty_mode"]

        # for rambo
        self.dynamics_update_freq = 0 if "dynamics_update_freq" not in self.args else self.args["dynamics_update_freq"]

    def set_data_buffer(self, buffer):
        real_buffer = ReplayBuffer(buffer_size=len(buffer["obs"]),
                                   obs_shape=self.args['obs_shape'],
                                   obs_dtype=np.float32,
                                   action_dim=self.action_dim,
                                   action_dtype=np.float32,
                                   device=self.args['device']
                                   )
        real_buffer.load_dataset(buffer)
        # self.obs_max = np.concatenate([real_buffer.observations, real_buffer.next_observations], axis=0).max(axis=0)
        # self.obs_min = np.concatenate([real_buffer.observations, real_buffer.next_observations], axis=0).min(axis=0)
        # self.rew_max = np.concatenate([real_buffer.rewards, real_buffer.rewards], axis=0).max(axis=0)
        # self.rew_min = np.concatenate([real_buffer.rewards, real_buffer.rewards], axis=0).min(axis=0)

        obs_mean, obs_std = real_buffer.normalize_obs(inplace=self.args['normalize_obs'])
        # fake_buffer = ReplayBuffer(buffer_size=int(self.fake_buffer_size), 
        #                            obs_shape=self.args['obs_shape'],
        #                            obs_dtype=np.float32,
        #                            action_dim=self.action_dim,
        #                            action_dtype=np.float32,
        #                            device=self.args['device']
        #                            )
        return real_buffer, obs_mean, obs_std

    def set_ensemble_dynamics(self, obs_mean, obs_std):
        termination_fn = obs_unnormalization(self._terminal_fn, obs_mean, obs_std) if self.args["normalize_obs"] else self._terminal_fn
        policy_scaler = StandardScaler(mu=obs_mean, std=obs_std)
        dynamics = EnsembleDynamics(self.dynamics_model,
                                    self.dynamics_optim,
                                    self.dynamics_scaler,
                                    termination_fn,
                                    penalty_coef=self.penalty_coef,
                                    uncertainty_mode=self.uncertainty_mode,
                                    data_range=(self.obs_min, self.obs_max, self.rew_min, self.rew_max),
                                    )
        return termination_fn, dynamics, policy_scaler

    def train(self, train_buffer, val_buffer, callback_fn):
        total_buffer = train_buffer
        if val_buffer is not None:
            for k, v in train_buffer.items():
                total_buffer[k] = np.concatenate([total_buffer[k], val_buffer[k]], axis=0)

        self.obs_max = train_buffer['obs'].max(axis=0)
        self.obs_min = train_buffer['obs'].min(axis=0)
        expert_range = self.obs_max - self.obs_min
        soft_expanding = expert_range * 0.05
        self.obs_max += soft_expanding
        self.obs_min -= soft_expanding
        # self.obs_max = np.maximum(self.obs_max, 100)
        # self.obs_min = np.minimum(self.obs_min, -100)

        self.rew_max = train_buffer['rew'].max()
        self.rew_min = train_buffer['rew'].min() - self.args['penalty_clip'] * self.args['penalty_coef']

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
            self.train_transition(self.real_buffer.sample_all())
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

        env_pool_size = int((train_buffer.shape[0]/self.args['horizon']) * 1.2)
        self.env_pool = SimpleReplayTrajPool(self.obs_space, self.action_space, self.args['horizon'],\
                                             self.args['lstm_hidden_unit'], env_pool_size)
        self.model_pool = SimpleReplayTrajPool(self.obs_space, self.action_space, self.args['horizon'],\
                                               self.args['lstm_hidden_unit'],self.args['model_pool_size'])

        loader.restore_pool_d4rl(self.env_pool, self.args['data_name'], adapt=True,\
                                 maxlen=self.args['horizon'],policy_hook=self.policy_gru,\
                                 value_hook=self.value_gru, device=self.device)
        torch.cuda.empty_cache()
        # self.obs_max = train_buffer['obs'].max(axis=0)
        # self.obs_min = train_buffer['obs'].min(axis=0)
        # expert_range = self.obs_max - self.obs_min
        # soft_expanding = expert_range * 0.05
        # self.obs_max += soft_expanding
        # self.obs_min -= soft_expanding
        # # self.obs_max = np.maximum(self.obs_max, 100)
        # # self.obs_min = np.minimum(self.obs_min, -100)

        # self.rew_max = train_buffer['rew'].max()
        # self.rew_min = train_buffer['rew'].min() - self.args['penalty_clip'] * self.args['lam']

        for i in range(self.args['out_train_epoch']):
            uncertainty_mean, uncertainty_max = self.rollout_model(self.args['rollout_batch_size'])
            torch.cuda.empty_cache()

            train_loss = {}
            train_loss['policy_loss'] = 0
            train_loss['q_loss'] = 0
            train_loss['uncertainty_mean'] = uncertainty_mean
            train_loss['uncertainty_max'] = uncertainty_max
            for j in range(self.args['in_train_epoch']):
                batch = self.get_train_policy_batch(self.args['train_batch_size'])
                in_res = self.train_policy(batch)
                for key in in_res:
                    train_loss[key] = train_loss[key] + in_res[key]
            for k in train_loss:
                train_loss[k] = train_loss[k]/self.args['in_train_epoch']
            
            # evaluate in mujoco
            eval_loss = self.eval_policy()
            if i % 100 == 0 or i == self.args['out_train_epoch'] - 1:
                self.eval_one_trajectory()
            train_loss.update(eval_loss)
            torch.cuda.empty_cache()
            self.log_res(i, train_loss)
            if i%4 == 0:
                loader.reset_hidden_state(self.env_pool, self.args['data_name'],\
                                 maxlen=self.args['horizon'],policy_hook=self.policy_gru,\
                                 value_hook=self.value_gru, device=self.device)
            torch.cuda.empty_cache()

    def get_train_policy_batch(self, batch_size = None):
        batch_size = batch_size or self.args['train_batch_size']
        env_batch_size = int(batch_size * self.args['real_data_ratio'])
        model_batch_size = batch_size - env_batch_size

        env_batch = self.env_pool.random_batch(env_batch_size)

        if model_batch_size > 0:
            model_batch = self.model_pool.random_batch(model_batch_size)

            keys = set(env_batch.keys()) & set(model_batch.keys())
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch

    def get_policy(self):
        return self.policy_gru , self.actor

    def get_meta_action(self, state: np.ndarray, hidden: np.ndarray, deterministic=False, out_mean_std=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, dim=1)
        lens = [1] *state.shape[0]
        hidden_policy, lst_action = hidden
        if isinstance(lst_action, np.ndarray):
            lst_action = torch.from_numpy(lst_action).to(self.device)
        if isinstance(hidden_policy, np.ndarray):
            hidden_policy = torch.from_numpy(hidden_policy).to(self.device)

        if len(hidden_policy.shape) == 2:
            hidden_policy = torch.unsqueeze(hidden_policy, dim=0)
        if len(lst_action.shape) == 2:
            lst_action = torch.unsqueeze(lst_action, dim=1)
        
        hidden_policy_res = self.policy_gru(state,lst_action,hidden_policy, lens)
        mu_res, action_res, log_p_res, std_res = self.actor(hidden_policy_res, state)
        hidden_policy_res = torch.squeeze(hidden_policy_res, dim=1)
        action_res = torch.squeeze(action_res, dim=1)
        mu_res = torch.squeeze(mu_res, dim=1)
        std_res = torch.squeeze(std_res, dim=1)

        if out_mean_std:
            return mu_res, action_res, std_res, hidden_policy_res

        if deterministic:
            return mu_res, hidden_policy_res
        else:
            return action_res, hidden_policy_res
    
    def train_transition(self, data):
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
            new_val_losses = self.validate(val_inputs, val_targets)
            print(f"epoch: {epoch}, val loss: {new_val_losses}")

            val_loss = (np.sort(new_val_losses)[:self.dynamics.model.num_elites]).mean().item()
            self.log_res(epoch, {"loss/dynamics_train_loss": train_loss, "loss/dynamics_val_loss": val_loss})

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 1e-2:
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

    def rollout_model(self,rollout_batch_size, deterministic=False):
        batch = self.env_pool.random_batch_for_initial(rollout_batch_size)
        obs = batch['observations']
        lst_action = batch['last_actions']

        hidden_value_init = batch['value_hidden']
        hidden_policy_init = batch['policy_hidden']

        uncertainty_list = []
        uncertainty_max = []

        current_nonterm = np.ones((len(obs)), dtype=bool)
        samples = None
        with torch.no_grad():
            hidden_policy = hidden_policy_init
            hidden = (hidden_policy, lst_action)
            for i in range(self.args['horizon']):
                act, hidden_policy = self.get_meta_action(obs, hidden, deterministic)
                act = act.cpu().numpy()
                hidden_policy = hidden_policy.cpu().numpy()

                next_obs, penalized_reward, term, info = self.dynamics.step(obs, act, 
                                                                            transition_scaler=self.args['transition_scaler'], 
                                                                            transition_clip=self.args['trainsition_clip'])
                uncertainty_list.append(info["penalty"].mean().item())
                uncertainty_max.append(info["penalty"].max().item())
                nonterm_mask = ~term.squeeze(-1)
                print('average reward:', info["raw_reward"].mean().item())
                print('average uncertainty:', info["penalty"].mean().item())

                #nonterm_mask: 1-not done, 0-done
                samples = {'observations': obs, 'actions': act, 'next_observations': next_obs,
                           'rewards': penalized_reward, 'terminals': term,
                           'last_actions': lst_action,
                           'valid': current_nonterm.reshape(-1, 1),
                           'value_hidden': hidden_value_init, 'policy_hidden': hidden_policy_init} 
                samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
                num_samples = samples['observations'].shape[0]
                index = np.arange(self.model_pool._pointer, self.model_pool._pointer + num_samples) % self.model_pool._max_size
                for k in samples:
                    self.model_pool.fields[k][index, i] = samples[k][:, 0]
                current_nonterm = current_nonterm & nonterm_mask
                obs = next_obs
                lst_action = act
                hidden = (hidden_policy, lst_action)
            self.model_pool._pointer += num_samples
            self.model_pool._pointer %= self.model_pool._max_size
            self.model_pool._size = min(self.model_pool._max_size, self.model_pool._size + num_samples)
        return np.mean(uncertainty_list), np.max(uncertainty_max)


    def train_policy(self, batch):
        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][:,:max_len]).to(self.device)
        value_hidden = batch['value_hidden'][:,0]
        policy_hidden = batch['policy_hidden'][:,0]
        value_state = self.value_gru(batch['observations'], batch['last_actions'],value_hidden,lens)
        policy_state = self.policy_gru(batch['observations'], batch['last_actions'], policy_hidden, lens)
        lens_next = torch.ones(len(lens)).int()
        next_value_hidden = value_state[:,-1]
        next_policy_hidden = policy_state[:,-1]
        value_state_next = self.value_gru(batch['next_observations'][:,-1:],\
                                          batch['actions'][:,-1:],next_value_hidden,lens_next)
        policy_state_next = self.policy_gru(batch['next_observations'][:,-1:],\
                                            batch['actions'][:,-1:],next_policy_hidden,lens_next)

        value_state_next = torch.cat([value_state[:,1:],value_state_next],dim=1)
        policy_state_next = torch.cat([policy_state[:,1:],policy_state_next],dim=1)

        q1 = self.q1(value_state,batch['actions'],batch['observations'])
        q2 = self.q2(value_state,batch['actions'],batch['observations'])
        valid_num = torch.sum(batch['valid'])
        
        with torch.no_grad():
            mu_target, act_target, log_p_act_target, std_target = self.actor(policy_state_next,\
                                                                             batch['next_observations'])
            q1_target = self.target_q1(value_state_next,act_target,batch['next_observations'])
            q2_target = self.target_q2(value_state_next,act_target,batch['next_observations'])
            Q_target = torch.min(q1_target,q2_target)
            alpha = self._alpha
            Q_target = Q_target - alpha*torch.unsqueeze(log_p_act_target,dim=-1)
            Q_target = batch['rewards'] + self.args['discount']*(~batch['terminals'])*(Q_target)
            Q_target = torch.clamp(Q_target,self.rew_min/(1-self.args['discount']),\
                                  self.rew_max/(1-self.args['discount']))

        q1_loss = torch.sum(((q1-Q_target)**2)*batch['valid'])/valid_num
        q2_loss = torch.sum(((q2-Q_target)**2)*batch['valid'])/valid_num
        q_loss = (q1_loss+q2_loss)/2

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
        self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])

        mu_now, act_now, log_p_act_now, std_now = self.actor(policy_state, batch['observations'])
        log_p_act_now = torch.unsqueeze(log_p_act_now, dim=-1)

        if self.args['learnable_alpha']:
            # update alpha
            alpha_loss = - torch.sum(self._log_alpha * ((log_p_act_now + self._target_entropy)*batch['valid']).detach())/valid_num

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        q1_ = self.q1(value_state.detach(), act_now, batch['observations'])
        q2_ = self.q2(value_state.detach(), act_now, batch['observations'])
        min_q_ = torch.min(q1_, q2_)
        policy_loss = alpha*log_p_act_now - min_q_
        policy_loss = torch.sum(policy_loss*batch['valid'])/valid_num

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        res = {}
        res['policy_loss'] = policy_loss.cpu().detach().numpy()
        res['q_loss'] = q_loss.cpu().detach().numpy()
        return res

    # def _select_best_indexes(self, metrics, n):
    #     pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
    #     pairs = sorted(pairs, key=lambda x: x[0])
    #     selected_indexes = [pairs[i][1] for i in range(n)]
    #     return selected_indexes

    # def _train_transition(self, transition, data, optim):
    #     data.to_torch(device=self.device)
    #     ''' calculation in MOPO '''
    #     dist = transition(torch.cat([data['obs'], data['act']], dim=-1))
    #     loss = - dist.log_prob(torch.cat([data['obs_next'], data['rew']], dim=-1))
    #     loss = loss.sum()
    #     ''' calculation when not deterministic TODO: figure out the difference A: they are the same when Gaussian''' 
    #     loss += 0.01 * (2. * transition.max_logstd).sum() - 0.01 * (2. * transition.min_logstd).sum()

    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()

    # def _eval_transition(self, transition, valdata, inc_var_loss=True):
    #     with torch.no_grad():
    #         valdata.to_torch(device=self.device)
    #         dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
    #         if inc_var_loss:
    #             mse_losses = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2 / (dist.variance + 1e-8)).mean(dim=(1, 2))
    #             logvar = 2 * transition.max_logstd - torch.nn.functional.softplus(2 * transition.max_logstd - torch.log(dist.variance))
    #             logvar = 2 * transition.min_logstd + torch.nn.functional.softplus(logvar - 2 * transition.min_logstd)
    #             var_losses = logvar.mean(dim=(1, 2))
    #             loss = mse_losses + var_losses
    #         else:
    #             loss = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1, 2))
    #         return loss

    def eval_policy(self):
        env = get_env(self.args['task'])
        eval_res = OrderedDict()
        res = self.test_on_real_env(self.args['number_runs_eval'], env)
        return res

    def test_on_real_env(self, number_runs, env):
        results = ([self.test_one_trail(env) for _ in range(number_runs)])
        rewards = [result[0] for result in results]
        episode_lengths = [result[1] for result in results]

        rew_mean = np.mean(rewards)
        len_mean = np.mean(episode_lengths)

        res = OrderedDict()
        res["Reward_Mean_Env"] = rew_mean
        try:
            res["Score"] = env.get_normalized_score(rew_mean)
        except AttributeError:
            pass
        res["Length_Mean_Env"] = len_mean

        return res

    def test_one_trail(self, env):
        env = deepcopy(env)
        with torch.no_grad():
            state, done = env.reset(), False
            if isinstance(state, tuple):
                # neorl2 reset returns a tuple
                state = state[0]
            lst_action = torch.zeros((1,1,self.action_dim)).to(self.device)
            hidden_policy = torch.zeros((1,1,self.args['lstm_hidden_unit'])).to(self.device)
            rewards = 0
            lengths = 0
            while not done:
                state = state[np.newaxis]  
                state = torch.from_numpy(state).float().to(self.device)
                hidden = (hidden_policy, lst_action)
                action, hidden_policy = self.get_meta_action(state, hidden, deterministic=True)
                use_action = action.cpu().numpy().reshape(-1)
                result = env.step(use_action)
                if len(result) == 4:
                    state_next, reward, done, _ = result
                else:
                    state_next, reward, done, timeout,_ = result
                    done = done or timeout
                lst_action = action
                state = state_next
                rewards += reward
                lengths += 1
        return (rewards, lengths)

    # to check the mean of actions
    def eval_one_trajectory(self):
        env = get_env(self.args['task'])
        env = deepcopy(env)
        with torch.no_grad():
            state, done = env.reset(), False
            if isinstance(state, tuple):
                # neorl2 reset returns a tuple
                state = state[0]
            lst_action = torch.zeros((1,1,self.action_dim)).to(self.device)
            hidden_policy = torch.zeros((1,1,self.args['lstm_hidden_unit'])).to(self.device)
            rewards = 0
            lengths = 0
            mu_list = []
            std_list = []
            while not done:
                state = state[np.newaxis]
                state = torch.from_numpy(state).float().to(self.device)
                hidden = (hidden_policy, lst_action)
                mu, action, std, hidden_policy = self.get_meta_action(state, hidden, deterministic=True, out_mean_std=True)
                mu_list.append(mu.cpu().numpy())
                std_list.append(std.cpu().numpy())
                use_action = action.cpu().numpy().reshape(-1)
                result = env.step(use_action)
                if len(result) == 4:
                    state_next, reward, done, _ = result
                else:
                    state_next, reward, done, timeout,_ = result
                    done = done or timeout
                lst_action = action
                state = state_next
                rewards += reward
                lengths += 1
            print("======== Action Mean mean: {}, Action Std mean: {}, Reward: {}, Length: {} ========"\
                .format(np.mean(np.array(mu_list), axis=0), np.mean(np.array(std_list), axis=0), reward, lengths))
            
