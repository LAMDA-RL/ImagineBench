# MOBILE: Model-Bellman Inconsistency Penalized Policy Optimization
# https://proceedings.mlr.press/v202/sun23q.html
# https://github.com/yihaosun1124/mobile

import torch
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.new_ensemble import EnsembleTransition, StandardScaler
from offlinerl.utils.net.terminal_check import get_termination_fn
from offlinerl.utils.env import get_env_obs_act_spaces

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
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args['device'])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    critics = []
    for i in range(args['num_q_ensemble']):
        critic_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=args['hidden_dims'])
        critics.append(Critic(critic_backbone, args['device']))
    critics = torch.nn.ModuleList(critics)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args['critic_lr'])

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
        "critic" : {"net" : critics, "opt" : critics_optim},
    }


class AlgoTrainer(ModelBasedAlgoTrainer):
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

        self.critics = algo_init['critic']['net']
        self.target_critics = deepcopy(self.critics)
        self.target_critics.eval()
        self.critics_optim = algo_init['critic']['opt']

        self.device = args['device']
        self._tau = self.args['soft_target_tau']
        self._gamma = self.args['discount']
        
        self.fake_buffer_size = self.args["model_retain_epochs"] * self.args["rollout_batch_size"] * self.args["horizon"]
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim, args['max_epoch'])

        self.deterministic_backup = True
        self.transition_clip = self.args["trainsition_clip"]

        # for mopo
        self.penalty_coef = 0.0 if "penalty_coef" not in self.args else self.args["penalty_coef"]
        self.uncertainty_mode = "None" if "uncertainty_mode" not in self.args else self.args["uncertainty_mode"]

        # for rambo
        self.dynamics_update_freq = 0 if "dynamics_update_freq" not in self.args else self.args["dynamics_update_freq"]

    def train_mode(self) -> None:
        self.actor.train()
        self.critics.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critics.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.target_critics.parameters(), self.critics.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        
    @ torch.no_grad()
    def compute_lcb(self, obss: torch.Tensor, actions: torch.Tensor):
        # compute next q std
        pred_next_obss = self.dynamics.sample_next_obss(obss, actions, self.args['num_samples'], transition_scaler=self.args['transition_scaler'], transition_clip=self.transition_clip)
        num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
        pred_next_obss = pred_next_obss.reshape(-1, obs_dim)
        pred_next_actions, _ = self.actforward(pred_next_obss)
        
        pred_next_qs =  torch.cat([target_critic(pred_next_obss, pred_next_actions) for target_critic in self.target_critics], 1)
        pred_next_qs = torch.min(pred_next_qs, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
        penalty = pred_next_qs.mean(0).std(0)

        return penalty
    
    def policy_learn(self, batch):
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        
        obss, actions, next_obss, rewards, terminals = \
            mix_batch["observations"], mix_batch["actions"], mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]

        # update critic
        qs = torch.stack([critic(obss, actions) for critic in self.critics], 0)
        with torch.no_grad():
            penalty = self.compute_lcb(obss, actions)
            penalty_clip = max(self.rew_max[0] - self.rew_min[0], self.rew_max[0])
            penalty = torch.clamp(penalty, 0, penalty_clip)
            penalty[:len(real_batch["rewards"])] = 0.0

            next_actions, next_log_probs = self.actforward(next_obss)
            next_qs = torch.cat([target_critic(next_obss, next_actions) for target_critic in self.target_critics], 1)
            next_q = torch.min(next_qs, 1)[0].reshape(-1, 1)
            if not self.deterministic_backup:
                next_q -= self._alpha * next_log_probs
            target_q = (rewards - self.args['penalty_coef'] * penalty) + self._gamma * (1 - terminals) * next_q
            # target_q = torch.clamp(target_q, 0, None)

        critic_loss = ((qs - target_q) ** 2).mean()
        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        qas = torch.cat([critic(obss, a) for critic in self.critics], 1)
        actor_loss = -torch.min(qas, 1)[0].mean() + self._alpha * log_probs.mean()
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
            "loss/critic": critic_loss.item(),
            "penalty": penalty.mean().item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result
