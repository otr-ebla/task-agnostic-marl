#!/usr/bin/env python3
"""Entry point: train MAPPO agents on the indoor coverage task."""

import argparse
import csv
import os

import numpy as np
import torch

from src.envs.coverage_vector_env import MultiRobotCoverageEnv
from src.models.actor_critic_mlp import Actor, Critic
from src.algorithms.mappo import MAPPO, RolloutBuffer
from src.utils.config_parser import load_config


# ---------------------------------------------------------------------------
# Observation normalisation (Welford online algorithm)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Tracks running mean and variance for online observation normalisation."""

    def __init__(self, shape: tuple, eps: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        # x : (B, *shape) — a batch of observations
        x = x.reshape(-1, *self.mean.shape)
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean  = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var   = M2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state_dict(self) -> dict:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean  = d['mean']
        self.var   = d['var']
        self.count = d['count']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_lr_decay(
    optimizer: torch.optim.Optimizer,
    initial_lr: float,
    current_update: int,
    total_updates: int,
) -> None:
    frac = max(0.0, 1.0 - current_update / total_updates)
    for group in optimizer.param_groups:
        group['lr'] = initial_lr * frac


def save_checkpoint(
    path: str,
    update: int,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    mappo: MAPPO,
    obs_rms: RunningMeanStd | None,
) -> None:
    payload = {
        'update':           update,
        'actor_state':      actor.state_dict(),
        'critic_state':     critic.state_dict(),
        'actor_opt_state':  mappo.actor_opt.state_dict(),
        'critic_opt_state': mappo.critic_opt.state_dict(),
    }
    if obs_rms is not None:
        payload['obs_rms'] = obs_rms.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    mappo: MAPPO,
    obs_rms: RunningMeanStd | None,
) -> int:
    ckpt = torch.load(path, map_location='cpu')
    actor.load_state_dict(ckpt['actor_state'])
    critic.load_state_dict(ckpt['critic_state'])
    mappo.actor_opt.load_state_dict(ckpt['actor_opt_state'])
    mappo.critic_opt.load_state_dict(ckpt['critic_opt_state'])
    if obs_rms is not None and 'obs_rms' in ckpt:
        obs_rms.load_state_dict(ckpt['obs_rms'])
    return int(ckpt['update'])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str, save_dir: str, resume: str | None) -> tuple:
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    env_cfg   = config.get('env',   {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    env = MultiRobotCoverageEnv(env_cfg)
    N          = env.num_robots
    obs_dim    = env.obs_dim
    state_dim  = env.state_dim
    action_dim = 2

    print(f"obs_dim={obs_dim}  state_dim={state_dim}  "
          f"grid={env.grid_w}×{env.grid_h}  robots={N}  "
          f"soft_collision={not env.terminate_on_collision}  "
          f"local_coverage={env.use_local_coverage_obs}")

    hidden_size   = model_cfg.get('hidden_size',   128)
    gru_hidden    = model_cfg.get('gru_hidden',    128)
    critic_hidden = model_cfg.get('critic_hidden', 256)

    actor  = Actor(obs_dim, action_dim, hidden_size, gru_hidden)
    critic = Critic(state_dim, critic_hidden)

    mappo = MAPPO(actor, critic, train_cfg, device=str(device))

    T             = train_cfg.get('rollout_steps',  256)
    total_updates = train_cfg.get('total_updates',  3000)
    log_interval  = train_cfg.get('log_interval',   10)
    save_interval = train_cfg.get('save_interval',  100)
    gamma         = train_cfg.get('gamma',          0.99)
    gae_lambda    = train_cfg.get('gae_lambda',     0.95)
    normalize_obs = train_cfg.get('normalize_obs',  True)
    lr_decay      = train_cfg.get('lr_decay',       True)
    lr_actor_0    = train_cfg.get('lr_actor',       3e-4)
    lr_critic_0   = train_cfg.get('lr_critic',      1e-3)

    buffer = RolloutBuffer(T, N, obs_dim, state_dim, action_dim, gru_hidden,
                           torch.device(str(device)))

    obs_rms = RunningMeanStd(shape=(obs_dim,)) if normalize_obs else None

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'training_log.csv')

    start_update = 1
    if resume:
        start_update = load_checkpoint(resume, actor, critic, mappo, obs_rms) + 1
        print(f"Resumed from {resume}, continuing at update {start_update}")

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['update', 'episodes', 'mean_ep_reward',
                         'coverage_ratio', 'actor_loss', 'critic_loss', 'entropy'])

    # -- Initial env state --
    obs, info  = env.reset()
    state      = env.get_global_state()
    hx         = actor.init_hidden(N, device=str(device))

    ep_rewards: list[float] = []
    ep_reward   = 0.0
    ep_count    = 0
    last_info   = info

    for update in range(start_update, total_updates + 1):

        if lr_decay:
            linear_lr_decay(mappo.actor_opt,  lr_actor_0,  update, total_updates)
            linear_lr_decay(mappo.critic_opt, lr_critic_0, update, total_updates)

        # ------------------------------------------------------------
        # Collect T environment steps
        # ------------------------------------------------------------
        for _ in range(T):
            obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
            if obs_rms is not None:
                obs_rms.update(obs)

            actions_np, log_probs, value, new_hx = mappo.get_actions(obs_in, state, hx)

            next_obs, rewards, terminated, truncated, last_info = env.step(actions_np)
            done = terminated or truncated

            buffer.insert(
                torch.from_numpy(obs_in).float().to(device),
                torch.from_numpy(state).float().to(device),
                torch.from_numpy(actions_np).float().to(device),
                log_probs.to(device),
                torch.from_numpy(rewards).float().to(device),
                value.to(device),
                terminated,
                hx,
            )

            ep_reward += float(rewards.mean())
            obs   = next_obs
            state = env.get_global_state()
            hx    = new_hx if not done else actor.init_hidden(N, device=str(device))

            if done:
                ep_rewards.append(ep_reward)
                ep_count  += 1
                ep_reward  = 0.0
                obs, _     = env.reset()
                state      = env.get_global_state()

        # ------------------------------------------------------------
        # Bootstrap value for the state after the last collected step
        # ------------------------------------------------------------
        obs_in_last = obs_rms.normalize(obs) if obs_rms is not None else obs
        _, _, last_value, _ = mappo.get_actions(obs_in_last, state, hx)
        buffer.compute_returns(last_value.to(device), gamma, gae_lambda)

        # ------------------------------------------------------------
        # Policy update
        # ------------------------------------------------------------
        losses = mappo.update(buffer)

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------
        if update % log_interval == 0:
            window    = ep_rewards[-20:] if ep_rewards else [0.0]
            mean_ep_r = float(np.mean(window))
            print(
                f"Update {update:5d}/{total_updates} | "
                f"episodes={ep_count:5d} | "
                f"mean_ep_r={mean_ep_r:8.3f} | "
                f"coverage={last_info['coverage_ratio']:.2%} | "
                f"actor={losses['actor_loss']:7.4f} | "
                f"critic={losses['critic_loss']:7.4f} | "
                f"entropy={losses['entropy']:6.4f}"
            )
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    update, ep_count, round(mean_ep_r, 4),
                    round(last_info['coverage_ratio'], 4),
                    round(losses['actor_loss'],  4),
                    round(losses['critic_loss'], 4),
                    round(losses['entropy'],     4),
                ])

        if update % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f'checkpoint_{update:05d}.pt')
            save_checkpoint(ckpt_path, update, actor, critic, mappo, obs_rms)
            print(f"  → checkpoint saved: {ckpt_path}")

    # Final checkpoint
    save_checkpoint(os.path.join(save_dir, 'checkpoint_final.pt'),
                    total_updates, actor, critic, mappo, obs_rms)

    env.close()
    return actor, critic, obs_rms


if __name__ == '__main__':
    _default_cfg = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'mappo_baseline.yaml'
    )
    _default_save = os.path.join(
        os.path.dirname(__file__), '..', 'checkpoints'
    )
    parser = argparse.ArgumentParser(description='Train MAPPO for indoor coverage')
    parser.add_argument('--config',   default=_default_cfg,
                        help='Path to YAML config file')
    parser.add_argument('--save-dir', default=_default_save,
                        help='Directory for checkpoints and training log')
    parser.add_argument('--resume',   default=None,
                        help='Path to a checkpoint to resume from')
    args = parser.parse_args()
    train(args.config, args.save_dir, args.resume)
