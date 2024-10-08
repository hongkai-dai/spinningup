import spinup.algos.pytorch.vpg.my_vpg as mut
import numpy as np
import pytest
import torch


class TestVPG:
    @classmethod
    def setup_class(cls):
        cls.obs_dim = 4
        cls.act_dim = 2
        cls.T = 10
        cls.tv_actor = mut.MLPGaussianActor(
            cls.obs_dim,
            cls.act_dim,
            hidden_sizes=[4, 8, 4],
            activation=torch.nn.ReLU(),
            sigma=0.1,
            time_varying=True,
        )
        cls.ti_actor = mut.MLPGaussianActor(
            cls.obs_dim,
            cls.act_dim,
            hidden_sizes=[4, 8, 4],
            activation=torch.nn.ReLU(),
            sigma=0.1,
            time_varying=False,
        )
        cls.tv_critic = mut.MLPCritic(
            cls.obs_dim,
            hidden_sizes=[4, 8, 16],
            activation=torch.nn.ReLU(),
            time_varying=True,
        )
        cls.ti_critic = mut.MLPCritic(
            cls.obs_dim,
            hidden_sizes=[4, 8, 16],
            activation=torch.nn.ReLU(),
            time_varying=False,
        )

    def test_vpg_cost(self):
        batch_size = 8
        T = 16
        obs = torch.rand(batch_size, T, self.obs_dim)
        t = torch.arange(0, T).to(torch.float)
        (act_ti, log_pi_ti) = self.ti_actor(obs.view((batch_size * T, self.obs_dim)))
        act_ti = act_ti.view((batch_size, T, self.act_dim))
        log_pi_ti = log_pi_ti.view((batch_size, T))
        (act_tv, log_pi_tv) = self.tv_actor(
            obs.view((batch_size * T, self.obs_dim)), t=t.repeat(batch_size)
        )
        act_tv = act_tv.view((batch_size, T, self.act_dim))
        log_pi_tv = log_pi_tv.view((batch_size, T))
        rew_ti = ((obs**2).sum(dim=-1) + (act_ti**2).sum(dim=-1)).detach()
        rew_tv = ((obs**2).sum(dim=-1) + (act_tv**2).sum(dim=-1)).detach()
        gamma = 0.95
        cost_ti = mut.vpg_cost(
            self.ti_actor,
            obs,
            act_ti,
            log_pi_ti,
            rew_ti,
            self.ti_critic,
            gamma,
            time_varying=False,
        )
        cost_tv = mut.vpg_cost(
            self.tv_actor,
            obs,
            act_tv,
            log_pi_tv,
            rew_tv,
            self.tv_critic,
            gamma,
            time_varying=True,
        )

        # Now compute the cost with for loop.
        t = torch.arange(0, T).to(torch.float)
        rtg_ti = torch.zeros((batch_size, T))
        rtg_tv = torch.zeros((batch_size, T))
        for i in range(batch_size):
            rtg_ti[i, -1] = rew_ti[i, -1] * gamma ** (T - 1)
            rtg_tv[i, -1] = rew_tv[i, -1] * gamma ** (T - 1)
            for j in range(T - 1, 0, -1):
                rtg_ti[i, j - 1] = rtg_ti[i, j] + rew_ti[i, j - 1] * gamma ** (j - 1)
                rtg_tv[i, j - 1] = rtg_tv[i, j] + rew_tv[i, j - 1] * gamma ** (j - 1)
        V_val_ti = torch.zeros((batch_size, T))
        V_val_tv = torch.zeros((batch_size, T))
        for i in range(batch_size):
            for j in range(T):
                V_val_ti[i, j] = self.ti_critic(obs[i, j])
                V_val_tv[i, j] = self.tv_critic(
                    obs[i, j], t=torch.tensor([j], dtype=torch.float)
                )
        cost_ti_expected = (log_pi_ti * (rtg_ti - V_val_ti)).sum()
        cost_tv_expected = (log_pi_tv * (rtg_tv - V_val_tv)).sum()
        np.testing.assert_allclose(
            cost_ti.detach().numpy(), cost_ti_expected.detach().numpy(), rtol=1e-5
        )
        np.testing.assert_allclose(
            cost_tv.detach().numpy(), cost_tv_expected.detach().numpy(), rtol=1e-5
        )
