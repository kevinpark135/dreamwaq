# test_cenet.py

import torch

from dreamwaq_cenet import DreamWaQCENet, cenet_loss


def main():
    num_envs = 64
    single_obs_dim = 45
    history_length = 5
    obs_history_dim = single_obs_dim * history_length

    model = DreamWaQCENet(
        obs_history_dim=obs_history_dim,
        single_obs_dim=single_obs_dim,
        latent_dim=16,
    )

    obs_history = torch.randn(num_envs, obs_history_dim)
    target_base_lin_vel = torch.randn(num_envs, 3)
    target_next_obs = torch.randn(num_envs, single_obs_dim)

    outputs = model(obs_history)
    losses = cenet_loss(outputs, target_base_lin_vel, target_next_obs, beta=1.0)

    print("v_hat:", outputs["v_hat"].shape)
    print("z:", outputs["z"].shape)
    print("obs_recon:", outputs["obs_recon"].shape)
    print("total_loss:", losses["total_loss"].item())
    print("velocity_loss:", losses["velocity_loss"].item())
    print("reconstruction_loss:", losses["reconstruction_loss"].item())
    print("kl_loss:", losses["kl_loss"].item())


if __name__ == "__main__":
    main()