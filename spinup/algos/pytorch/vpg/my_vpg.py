from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions


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


def vpg_cost(
    actor: Actor,
    obs: torch.Tensor,
    act: torch.Tensor,
    log_pi: torch.Tensor,
    rew: torch.Tensor,
    critic: Critic,
    gamma: float,
    time_varying: bool,
):
    """
    Compute the term ∑ₜlog π(aₜ|sₜ) * Gₜ
    where Gₜ is the advantage function ∑ₜ₋ₚᵣᵢₘₑ γᵗ⁻ᵖʳⁱᵐᵉ*rₜ₋ₚᵣᵢₘₑ − V(sₜ, t)
    obs: (batch_size, T, obs_dim)
    act: (batch_size, T, act_dim)
    rew: (batch_size, T)
    """
    batch_size = obs.shape[0]
    T = obs.shape[1]
    assert act.shape[0] == rew.shape[0] == batch_size
    assert act.shape[1] == rew.shape[1] == T
    assert log_pi.shape == (batch_size, T)
    assert rew.shape == (batch_size, T)
    assert len(rew.shape) == 2
    assert gamma > 0 and gamma <= 1
    t = torch.arange(0, T).to(torch.float)
    discounted_rew = torch.pow(gamma, t) * rew
    # cum_rew is of size (batch_size, T)
    cum_rew = discounted_rew.flip(1).cumsum(1).flip(1)
    obs_flat = obs.view((batch_size * T, -1))
    t_flat = t.unsqueeze(0).repeat((batch_size, 1)).view((-1,))
    V_val = critic(obs_flat, t=t_flat if time_varying else None).view((batch_size, T))
    cost = (log_pi * (cum_rew - V_val)).sum()
    return cost
