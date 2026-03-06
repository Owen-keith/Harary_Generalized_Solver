import os
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs.i4_env_vec import I4EnvVec
from models.resnet_policy_value import PolicyValueNet
from eval_i4 import evaluate_selfplay
from rl.rollout import collect_rollout
from rl.ppo import PPOConfig, ppo_update
from utils.tactics_i4 import generate_tactic_batch_from_env


@dataclass
class TrainConfig:
    # Env / rollout
    n_envs: int = 4096
    rollout_len: int = 64

    # PPO (more aggressive)
    ppo_epochs: int = 4
    minibatch_size: int = 8192
    lr: float = 1e-3

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.3
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 1.0

    # Aux tactics
    aux_coef: float = 0.1
    aux_batch_size: int = 2048
    tactic_sample_attempts: int = 8192

    # Training time
    updates: int = 300

    # Logging / checkpoint
    log_every: int = 1
    ckpt_every: int = 10

    # Eval
    eval_every: int = 10
    eval_games: int = 1024
    eval_batch_envs: int = 256

    seed: int = 0
    device: str = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    ap.add_argument("--run_name", type=str, default=None, help="Optional override for run name")
    args = ap.parse_args()

    cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed + 123)

    device = cfg.device if torch.cuda.is_available() else "cpu"

    os.makedirs("checkpoints", exist_ok=True)

    env = I4EnvVec(n_envs=cfg.n_envs, check_legal=False, seed=cfg.seed)
    env.reset()

    net = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    start_update = 1
    global_step = 0

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_update = int(ckpt.get("update", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))

        run_name = args.run_name or ckpt.get("run_name") or os.path.splitext(os.path.basename(args.resume))[0]
        print(f"Resuming from {args.resume}")
        print(f"Starting at update {start_update}, global_step {global_step}, run_name {run_name}")
    else:
        run_name = args.run_name or f"i4_ppo_{int(time.time())}"
        print(f"Starting new run: {run_name}")

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    ppo_cfg = PPOConfig(
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_eps=cfg.clip_eps,
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef,
        max_grad_norm=cfg.max_grad_norm,
        lr=cfg.lr,
        epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        aux_coef=cfg.aux_coef,
        aux_batch_size=cfg.aux_batch_size,
    )

    update = start_update - 1

    try:
        for update in range(start_update, cfg.updates + 1):
            t0 = time.time()

            batch = collect_rollout(
                env=env,
                net=net,
                device=device,
                rollout_len=cfg.rollout_len,
                gamma=cfg.gamma,
            )

            aux = generate_tactic_batch_from_env(
                env=env,
                n_samples=cfg.tactic_sample_attempts,
                rng=np_rng
            )

            metrics = ppo_update(
                net=net,
                optimizer=optimizer,
                batch=batch,
                device=device,
                cfg=ppo_cfg,
                aux_batch=aux,
            )

            total_env_steps = cfg.n_envs * cfg.rollout_len
            global_step += total_env_steps

            dt = time.time() - t0
            sps = total_env_steps / max(dt, 1e-6)

            if update % cfg.log_every == 0:
                writer.add_scalar("perf/steps_per_sec", sps, update)
                for k, v in metrics.items():
                    writer.add_scalar(k, v, update)

                aux_ce = metrics.get("loss/aux_ce", None)
                aux_sz = metrics.get("stats/aux_batch_size", None)
                aux_str = ""
                if aux_ce is not None:
                    aux_str = f" | aux_ce {aux_ce:.3f} | aux_b {aux_sz}"

                print(
                    f"update {update:5d} | sps {sps:,.0f} | "
                    f"ret {metrics['stats/return_mean']:.3f} | "
                    f"ent {metrics['stats/entropy']:.3f} | "
                    f"kl {metrics['stats/approx_kl']:.8e}"
                    f"{aux_str}"
                )

            if update % cfg.ckpt_every == 0:
                ckpt_path = os.path.join("checkpoints", f"{run_name}_u{update}.pt")
                torch.save(
                    {
                        "run_name": run_name,
                        "update": update,
                        "global_step": global_step,
                        "model_state": net.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "train_cfg": cfg.__dict__,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            if update % cfg.eval_every == 0:
                eval_metrics = evaluate_selfplay(
                    net=net,
                    device=device,
                    n_games=cfg.eval_games,
                    batch_envs=cfg.eval_batch_envs,
                    greedy=False,
                )
                for k, v in eval_metrics.items():
                    writer.add_scalar(k, v, update)

                print(
                    f"eval @ {update}: "
                    f"p1 {eval_metrics['eval/p1_win_rate']:.3f} "
                    f"p2 {eval_metrics['eval/p2_win_rate']:.3f} "
                    f"draw {eval_metrics['eval/draw_rate']:.3f} "
                    f"len {eval_metrics['eval/avg_len']:.2f}"
                )

    except KeyboardInterrupt:
        print("\nCaught Ctrl+C — saving an interrupt checkpoint...")

        ckpt_path = os.path.join("checkpoints", f"{run_name}_INTERRUPT.pt")
        torch.save(
            {
                "run_name": run_name,
                "update": update,
                "global_step": global_step,
                "model_state": net.state_dict(),
                "optim_state": optimizer.state_dict(),
                "train_cfg": cfg.__dict__,
            },
            ckpt_path,
        )
        print(f"Saved interrupt checkpoint: {ckpt_path}")

    finally:
        writer.close()


if __name__ == "__main__":
    main()