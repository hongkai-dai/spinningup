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

    def test_compute_rtg(self):
        batch_size = 4
        T = 10
        rew_path = torch.randn((batch_size, T))
        gamma = 0.95
        rtg = mut.compute_rtg(gamma, rew_path)
        assert rtg.shape == (batch_size, T)
        for i in range(batch_size):
            np.testing.assert_almost_equal(
                rtg[i, -1], gamma ** (T - 1) * rew_path[i, -1], decimal=5
            )
            for j in range(T - 1, 0, -1):
                np.testing.assert_almost_equal(
                    rtg[i, j - 1],
                    rtg[i, j] + gamma ** (j - 1) * rew_path[i, j - 1],
                    decimal=5,
                )

    def test_vpg_cost(self):
        batch_size = 8
        obs = torch.rand(batch_size, self.obs_dim)
        T = 10
        t = torch.floor(torch.rand((batch_size,)) * T).to(torch.float)
        (act_ti, log_pi_ti) = self.ti_actor(obs)
        (act_tv, log_pi_tv) = self.tv_actor(obs, t=t)
        rtg_ti = torch.rand((batch_size,))
        rtg_tv = torch.rand((batch_size,))

        cost_ti = mut.vpg_cost(
            obs,
            act_ti,
            log_pi_ti,
            rtg_ti,
            t,
            self.ti_critic,
            time_varying=False,
        )
        cost_tv = mut.vpg_cost(
            obs,
            act_tv,
            log_pi_tv,
            rtg_tv,
            t,
            self.tv_critic,
            time_varying=True,
        )

        # Now compute the cost with for loop.
        V_val_ti = self.ti_critic(obs)
        V_val_tv = self.tv_critic(obs, t=t)
        cost_ti_expected = (log_pi_ti * (rtg_ti - V_val_ti)).sum()
        cost_tv_expected = (log_pi_tv * (rtg_tv - V_val_tv)).sum()
        np.testing.assert_allclose(
            cost_ti.detach().numpy(), cost_ti_expected.detach().numpy(), rtol=1e-5
        )
        np.testing.assert_allclose(
            cost_tv.detach().numpy(), cost_tv_expected.detach().numpy(), rtol=1e-5
        )
