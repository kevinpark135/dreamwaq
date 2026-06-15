# dreamwaq_runner.py
"""DreamWaQ PPO runner with CENet integration."""

from __future__ import annotations

import os
import torch
from rsl_rl.runners import OnPolicyRunner

from .dreamwaq_cenet import DreamWaQCENet, cenet_loss


class DreamWaQRunner(OnPolicyRunner):
    """OnPolicyRunner extended with CENet update step."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # obs dim 확인
        single_obs_dim = env.observation_manager.group_obs_dim["policy"][0]
        history_length = getattr(env.cfg, "obs_history_length", 5)
        obs_history_dim = single_obs_dim * history_length

        # CENet 초기화
        self.cenet = DreamWaQCENet(
            obs_history_dim=obs_history_dim,
            single_obs_dim=single_obs_dim,
            latent_dim=16,
            hidden_dims=(128, 64),
        ).to(device)

        self.cenet_optimizer = torch.optim.Adam(
            self.cenet.parameters(), lr=1e-3
        )

        self.single_obs_dim = single_obs_dim

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Learn with CENet update after each PPO update."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            import time
            start = time.time()

            # obs_history 수집용 버퍼
            obs_history_buf = []
            next_obs_buf = []
            base_lin_vel_buf = []

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # obs_history 저장 (CENet 학습용)
                    obs_history_buf.append(
                        self.env.obs_history.clone()
                    )

                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )

                    # next obs, ground truth velocity 저장
                    next_obs_buf.append(obs[:, :self.single_obs_dim].clone())
                    base_lin_vel_buf.append(
                        self.env.scene["robot"].data.root_lin_vel_b.clone()
                    )

                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)

                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            # PPO 업데이트
            loss_dict = self.alg.update()

            # CENet 업데이트
            obs_history_tensor = torch.stack(obs_history_buf).reshape(
                -1, self.env.obs_history.shape[-1]
            ).to(self.device)
            next_obs_tensor = torch.stack(next_obs_buf).reshape(
                -1, self.single_obs_dim
            ).to(self.device)
            base_lin_vel_tensor = torch.stack(base_lin_vel_buf).reshape(
                -1, 3
            ).to(self.device)

            cenet_out = self.cenet(obs_history_tensor)
            ce_loss = cenet_loss(
                cenet_out,
                target_base_lin_vel=base_lin_vel_tensor,
                target_next_obs=next_obs_tensor,
                beta=1.0,
            )

            self.cenet_optimizer.zero_grad()
            ce_loss["total_loss"].backward()
            self.cenet_optimizer.step()

            # loss_dict에 CENet loss 추가 (TensorBoard 로깅)
            loss_dict["CENet/total"] = ce_loss["total_loss"].item()
            loss_dict["CENet/velocity"] = ce_loss["velocity_loss"].item()
            loss_dict["CENet/reconstruction"] = ce_loss["reconstruction_loss"].item()
            loss_dict["CENet/kl"] = ce_loss["kl_loss"].item()

            learn_time = time.time() - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(
                self.logger.log_dir,
                f"model_{self.current_learning_iteration}.pt"
            ))
            self.logger.stop_logging_writer()

    def save(self, path, infos=None):
        """Save PPO + CENet weights."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["cenet"] = self.cenet.state_dict()
        saved_dict["cenet_optimizer"] = self.cenet_optimizer.state_dict()
        torch.save(saved_dict, path)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        """Load PPO + CENet weights."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        self.alg.load(loaded_dict, load_cfg, strict)
        if "cenet" in loaded_dict:
            self.cenet.load_state_dict(loaded_dict["cenet"])
        if "cenet_optimizer" in loaded_dict:
            self.cenet_optimizer.load_state_dict(loaded_dict["cenet_optimizer"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")