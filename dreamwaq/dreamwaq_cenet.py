# dreamwaq_cenet.py
"""CENet module for DreamWaQ.

This file defines the Context-Aided Estimator Network used by DreamWaQ.
CENet receives flattened temporal proprioceptive observation history and
jointly learns:
1. explicit base linear velocity estimation,
2. latent context representation through a beta-VAE,
3. next-observation reconstruction as an auxiliary task.

This module is currently standalone. It is not yet connected to the rsl_rl
PPO actor-critic training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DreamWaQCENet(nn.Module):
    """Minimal CENet module for DreamWaQ.

    Input:
        obs_history: flattened temporal proprioceptive observations

    Output:
        estimated base linear velocity v_hat: [num_envs, 3]
        latent context z: [num_envs, latent_dim]
        reconstructed next observation obs_recon: [num_envs, single_obs_dim]
    """

    def __init__(
        self,
        obs_history_dim: int,
        single_obs_dim: int,
        latent_dim: int = 16,
        hidden_dims: tuple[int, int] = (128, 64),
    ):
        super().__init__()

        self.obs_history_dim = obs_history_dim
        self.single_obs_dim = single_obs_dim
        self.latent_dim = latent_dim

        h1, h2 = hidden_dims

        # Shared encoder: o_t^H -> feature
        self.encoder = nn.Sequential(
            nn.Linear(obs_history_dim, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
        )

        # Body velocity head: feature -> v_hat
        self.velocity_head = nn.Linear(h2, 3)

        # VAE latent heads: feature -> mu, logvar
        self.mu_head = nn.Linear(h2, latent_dim)
        self.logvar_head = nn.Linear(h2, latent_dim)

        # Decoder: z -> reconstructed next observation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ELU(),
            nn.Linear(h2, h1),
            nn.ELU(),
            nn.Linear(h1, single_obs_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using the VAE reparameterization trick."""
        logvar = logvar.clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_history: torch.Tensor):
        feature = self.encoder(obs_history)

        v_hat = self.velocity_head(feature)

        mu = self.mu_head(feature)
        logvar = self.logvar_head(feature).clamp(-10.0, 10.0)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        obs_recon = self.decoder(z)

        return {
            "v_hat": torch.nan_to_num(v_hat, nan=0.0, posinf=1.0e6, neginf=-1.0e6),
            "z": torch.nan_to_num(z, nan=0.0, posinf=1.0e6, neginf=-1.0e6),
            "mu": mu,
            "logvar": logvar,
            "obs_recon": torch.nan_to_num(obs_recon, nan=0.0, posinf=1.0e6, neginf=-1.0e6),
        }


def cenet_loss(
    outputs: dict[str, torch.Tensor],
    target_base_lin_vel: torch.Tensor,
    target_next_obs: torch.Tensor,
    beta: float = 1.0,
    velocity_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
):
    """CENet loss.

    L_CE = w_v * L_est + w_rec * L_rec + beta * L_KL
    L_est = smooth L1(v_hat, v)
    L_rec = MSE(obs_recon, next_obs)
    L_KL = KL(q(z|obs_history) || N(0, I))
    """
    v_hat = outputs["v_hat"]
    obs_recon = outputs["obs_recon"]
    mu = outputs["mu"]
    logvar = outputs["logvar"].clamp(-10.0, 10.0)

    target_base_lin_vel = torch.nan_to_num(
        target_base_lin_vel,
        nan=0.0,
        posinf=1.0e6,
        neginf=-1.0e6,
    )
    target_next_obs = torch.nan_to_num(
        target_next_obs,
        nan=0.0,
        posinf=1.0e6,
        neginf=-1.0e6,
    )

    target_base_lin_vel = target_base_lin_vel.clamp(-5.0, 5.0)
    v_hat = v_hat.clamp(-5.0, 5.0)

    velocity_loss = F.smooth_l1_loss(v_hat, target_base_lin_vel, beta=0.2)
    reconstruction_loss = F.mse_loss(obs_recon, target_next_obs)

    kl_loss = -0.5 * torch.mean(
        torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    )

    total_loss = velocity_weight * velocity_loss + reconstruction_weight * reconstruction_loss + beta * kl_loss

    return {
        "total_loss": total_loss,
        "velocity_loss": velocity_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }
