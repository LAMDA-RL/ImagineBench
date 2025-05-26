import os
from copy import deepcopy
from collections import defaultdict
import json

import torch
import numpy as np
from loguru import logger
from operator import itemgetter
from gym import spaces
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
    if args['task'] == "mujoco":
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18 + 768,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
    elif args['task'] == "libero":
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44 + 768,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
    else:
        obs_space, action_space = get_env_obs_act_spaces(args['task'])
    # obs_space, action_space = args["obs_space"], args["action_space"]
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
    dynamics_scaler = StandardScaler()

    _termination_fn = get_termination_fn(task=args['task'])

    return {
        "task_info" : {"obs_shape" : obs_shape, "action_shape" : action_shape, "action_dim": action_dim,
                       "obs_space": obs_space, "action_space": action_space},
        "transition" : {"net" : dynamics_model, "opt" : dynamics_optim,# "adv_opt" : dynamics_adv_optim, 
                        "scaler": dynamics_scaler, "_terminal_fn" : _termination_fn},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "alpha" : {"net" : alpha, "opt" : alpha_optim},
        "critic" : {"net" : [critic1, critic2], "opt" : [critic1_optim, critic2_optim]},
    }


class ModelBasedAlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(ModelBasedAlgoTrainer, self).__init__(args)
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

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self._is_auto_alpha = False
        self.alpha_optim = algo_init['alpha']['opt']
        if self.alpha_optim is not None:
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = algo_init['alpha']['net']
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = algo_init['alpha']['net']

        self.critic1, self.critic2 = algo_init['critic']['net']
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic1.eval()
        self.target_critic2 = deepcopy(self.critic2)
        self.target_critic2.eval()
        self.critic1_optim, self.critic2_optim = algo_init['critic']['opt']

        self.device = args['device']
        self._tau = self.args['soft_target_tau']
        self._gamma = self.args['discount']
        
        self.fake_buffer_size = self.args["model_retain_epochs"] * self.args["rollout_batch_size"] * self.args["horizon"]
        self.lr_scheduler = None

        # for mopo
        self.penalty_coef = 0.0 if "penalty_coef" not in self.args else self.args["penalty_coef"]
        self.uncertainty_mode = "None" if "uncertainty_mode" not in self.args else self.args["uncertainty_mode"]

        # for rambo
        self.dynamics_update_freq = 0 if "dynamics_update_freq" not in self.args else self.args["dynamics_update_freq"]


    def train_mode(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

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
        fake_buffer = ReplayBuffer(buffer_size=int(self.fake_buffer_size), 
                                   obs_shape=self.args['obs_shape'],
                                   obs_dtype=np.float32,
                                   action_dim=self.action_dim,
                                   action_dtype=np.float32,
                                   device=self.args['device']
                                   )
        return real_buffer, fake_buffer, obs_mean, obs_std

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

        self.real_buffer, self.fake_buffer, obs_mean, obs_std = self.set_data_buffer(total_buffer)
        self.termination_fn, self.dynamics, self.policy_scaler = self.set_ensemble_dynamics(obs_mean, obs_std)
        if self.args["policy_scaler"]:
            self.actor.set_scaler(self.policy_scaler)

        if self.args['dynamics_path'] is not None:
            print(f"Load dynamic model from {self.args['dynamics_path']}")
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
        else:
            print(f"Train dynamic model.")
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
    
    def get_policy(self):
        return self.actor
    
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

    def _sync_weight(self) -> None:
        for o, n in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def actforward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple:
        if self.args["normalize_obs"] and self.args['policy_scaler']:
            pass
        elif self.args["normalize_obs"] and not self.args['policy_scaler']:
            obs = self.policy_scaler.inverse_transform(obs)
        elif not self.args["normalize_obs"] and self.args['policy_scaler']:
            obs = self.policy_scaler.transform(obs)
        else:
            pass
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()
    
    def train_policy(self, callback_fn):
        num_timesteps = 0
        for epoch in range(self.args['max_epoch']):

            self.train_mode()

            self.real_buffer.prepare_tensor()
            self.fake_buffer.prepare_tensor()

            for _ in range(self.args['steps_per_epoch']):
                transition_update_flag, collect_data_flag = False, False
                if num_timesteps % self.args['rollout_freq'] == 0:
                    # collect data
                    collect_data_flag = True
                    init_obss = self.real_buffer.sample(int(self.args['rollout_batch_size']))["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.rollout(init_obss, self.args['horizon'])
                    self.fake_buffer.add_batch(**rollout_transitions)
                    # self.log_res(epoch, rollout_info)

                    self.real_buffer.prepare_tensor()
                    self.fake_buffer.prepare_tensor()

                real_sample_size = int(self.args['policy_batch_size'] * self.args['real_data_ratio'])
                fake_sample_size = self.args['policy_batch_size'] - real_sample_size
                # real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                # fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                real_batch = self.real_buffer.sample_t(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample_t(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy_learn(batch)

                # update transition
                if self.dynamics_update_freq > 0 and (num_timesteps+1) % self.dynamics_update_freq == 0:
                    transition_update_flag = True
                    all_loss_info = self.update_transition(self.real_buffer)

                num_timesteps += 1
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            res = callback_fn(self.get_policy())
            res.update(loss)
            
            if collect_data_flag:
                res.update(rollout_info)
            if transition_update_flag:
                res.update(all_loss_info)

            self.log_res(epoch, res)

        return self.get_policy()

    def rollout(self, init_obss, horizon):
        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(horizon):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions, 
                                                                             transition_scaler=self.args['transition_scaler'],
                                                                             transition_clip=self.args['trainsition_clip'])
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
