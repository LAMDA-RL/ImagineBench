import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerl.outside_utils.dynamics import BaseDynamics
from offlinerl.outside_utils.utils.scaler import StandardScaler
from offlinerl.outside_utils.utils.logger import Logger
import warnings


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        data_range: tuple = None,
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.obs_min, self.obs_max, self.rew_min, self.rew_max = data_range

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        transition_scaler: bool = True,
        transition_clip: bool = False,
        clip_penalty: bool = False,
        max_penalty: float = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        if transition_scaler:
            obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        if transition_clip:
            next_obs = np.clip(next_obs, self.obs_min, self.obs_max)
            reward = np.clip(reward, self.rew_min, self.rew_max)

        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef > 0.0:
            norm_mean = mean
            norm_std = std
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(norm_std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = norm_mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = norm_mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                warnings.warn("Invalid uncertainty mode. No penalty applied!!!")
                penalty = np.zeros_like(reward).mean(1)

            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            if clip_penalty:
                penalty = np.clip(penalty, a_max=max_penalty)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, np.bool_(terminal), info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int,
        transition_scaler: bool = True,
        transition_clip: bool = False,
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        if transition_scaler:
            obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        if transition_clip:
            obs_min = torch.as_tensor(self.obs_min).to(next_obss.device)
            obs_max = torch.as_tensor(self.obs_max).to(next_obss.device)
            next_obss = torch.clamp(next_obss, obs_min, obs_max)
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
