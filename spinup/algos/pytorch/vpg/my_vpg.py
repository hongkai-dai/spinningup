from typing import List, Optional, Tuple

import gym
import numpy as np
import tqdm
import torch
import torch.distributions
from torch.utils.data import TensorDataset, DataLoader


class Actor(torch.nn.Module):
    def _distribution(self, obs: torch.Tensor, *, t: torch.Tensor = None):
        raise NotImplementedError

    def _log_prob(self, pi, act: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, obs: torch.Tensor, *, t: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self._distribution(obs, t=t)
        act = pi.sample()
        log_pi = self._log_prob(pi, act)
        return act, log_pi


def mlp(sizes: List[int], activation: torch.nn.Module):
    layers = [torch.nn.Linear(sizes[0], sizes[1])]
    for i in range(1, len(sizes) - 1):
        layers.append(activation)
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
    return torch.nn.Sequential(*layers)


class MLPGaussianActor(Actor):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        activation: torch.nn.Module,
        sigma: float,
        time_varying: bool = False,
    ):
        super().__init__()
        self.mlp = mlp(
            [obs_dim + (1 if time_varying else 0)] + hidden_sizes + [act_dim],
            activation,
        )
        assert sigma > 0
        self.log_sigma = torch.nn.Parameter(
            torch.as_tensor(np.log(sigma)), requires_grad=False
        )
        self.time_varying = time_varying

    def _distribution(self, obs: torch.Tensor, *, t: torch.Tensor = None):
        if self.time_varying:
            mu_input = torch.cat((obs, t.view((-1, 1))), dim=1)
        else:
            mu_input = obs
        mu = self.mlp(mu_input)
        return torch.distributions.Normal(mu, torch.exp(self.log_sigma))

    def _log_prob(self, pi, act: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(act).sum(axis=-1)


class Critic(torch.nn.Module):
    def forward(self, obs: torch.Tensor, *, t: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError

    def _to_dataset(
        self, obs: torch.Tensor, gamma: float, rew: torch.Tensor
    ) -> TensorDataset:
        raise NotImplementedError

    def train(
        self,
        obs: torch.Tensor,
        rew: torch.Tensor,
        gamma: float,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 100,
        batch_size: int = 128,
    ):
        adv_dataset = self._to_dataset(obs, gamma, rew)

        loader = DataLoader(adv_dataset, batch_size=batch_size)
        for epoch in tqdm.tqdm(range(num_epochs)):
            for training_data in loader:
                x, t, adv_target = training_data
                adv = self(x, t=t)
                loss = torch.nn.MSELoss()(adv, adv_target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


class MLPCritic(Critic):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: List[int],
        activation: torch.nn.Module,
        time_varying: bool = False,
    ):
        super().__init__()
        self.mlp = mlp(
            [obs_dim + (1 if time_varying else 0)] + hidden_sizes + [1],
            activation,
        )
        self.time_varying = time_varying

    def forward(self, obs: torch.Tensor, *, t: torch.Tensor = None) -> torch.Tensor:
        if self.time_varying:
            mlp_input = torch.cat((obs, t.view((-1,) * (obs.dim() - 1) + (1,))), dim=-1)
        else:
            mlp_input = obs
        return self.mlp(mlp_input)

    def _to_dataset(
        self, obs: torch.Tensor, gamma: float, rew: torch.Tensor, device="cuda"
    ) -> TensorDataset:
        batch_size = obs.shape[0]
        rtg = compute_rtg(gamma, rew).to(device)
        T = rew.shape[1]
        t = torch.arange(T).to(torch.float).to(devie)
        t_batch = t.unsqueeze(0).repeat((batch_size, 1))
        return TensorDataset(obs, t_batch, rtg)


def compute_rtg(gamma: float, rew_path: torch.Tensor) -> torch.Tensor:
    """
    Compute reward-to-go ∑ₜ₋ₚᵣᵢₘₑ γᵗ⁻ᵖʳⁱᵐᵉ*rₜ₋ₚᵣᵢₘₑ

    Args:
      rew_path: Size is (batch_size, T)
    Return:
      rtg: Size is (batch_size, T)
    """
    T = rew_path.shape[1]
    t = torch.arange(0, T).to(torch.float)
    discounted_rew = torch.pow(gamma, t) * rew_path
    # cum_rew is of size (batch_size, T)
    cum_rew = discounted_rew.flip(1).cumsum(1).flip(1)
    return cum_rew


def vpg_cost(
    obs: torch.Tensor,
    act: torch.Tensor,
    log_pi: torch.Tensor,
    rtg: torch.Tensor,
    t: torch.Tensor,
    critic: Critic,
    time_varying: bool,
):
    """
    Compute the term ∑ₜlog π(aₜ|sₜ) * Gₜ
    where Gₜ is the advantage function ∑ₜ₋ₚᵣᵢₘₑ γᵗ⁻ᵖʳⁱᵐᵉ*rₜ₋ₚᵣᵢₘₑ − V(sₜ, t)
    obs: (batch_size, obs_dim)
    act: (batch_size, act_dim)
    rtg: (batch_size,), reward-to-go ∑ₜ₋ₚᵣᵢₘₑ γᵗ⁻ᵖʳⁱᵐᵉ*rₜ₋ₚᵣᵢₘₑ
    t:   (batch_size,)
    """
    batch_size = obs.shape[0]
    assert act.shape[0] == rtg.shape[0] == batch_size
    assert log_pi.shape == (batch_size,)
    assert rtg.shape == (batch_size,)
    assert t.shape == (batch_size,)
    V_val = critic(obs, t=t if time_varying else None)
    cost = (log_pi * (rtg - V_val)).sum()
    return cost


class VPGBuffer:
    def __init__(self, T: int, obs_dim: int, act_dim: int, buffer_size: int):
        self.obs_buffer = np.zeros((buffer_size, T, obs_dim))
        self.act_buffer = np.zeros((buffer_size, T, act_dim, T))
        self.rew_buffer = np.zeros((buffer_size, T))
        self.logp_buffer = np.zeros((buffer_size, T))
