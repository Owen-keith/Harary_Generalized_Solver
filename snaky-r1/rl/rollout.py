import numpy as np
import torch
from dataclasses import dataclass

from envs.i4_env_vec import I4EnvVec
from models.resnet_policy_value import bitboards_to_tensor, masked_softmax


@dataclass
class RolloutBatch:
    # Stored as torch tensors (mostly on CPU) for PPO
    obs_me: torch.Tensor        # (T, N) int64 (bitboards)
    obs_opp: torch.Tensor       # (T, N) int64
    actions: torch.Tensor       # (T, N) int64
    logprobs: torch.Tensor      # (T, N) float32
    values: torch.Tensor        # (T, N) float32
    rewards: torch.Tensor       # (T, N) float32
    dones: torch.Tensor         # (T, N) bool
    legal_mask: torch.Tensor    # (T, N, A) bool

    last_me: torch.Tensor       # (N,) int64
    last_opp: torch.Tensor      # (N,) int64
    last_done: torch.Tensor     # (N,) bool
    last_legal_mask: torch.Tensor  # (N, A) bool


@torch.no_grad()
def collect_rollout(
    env: I4EnvVec,
    net: torch.nn.Module,
    device: str,
    rollout_len: int,
    gamma: float = 0.99,
) -> RolloutBatch:
    """
    Collects rollout_len steps from vectorized env using current policy.
    Returns a RolloutBatch suitable for PPO (GAE computed later).
    """

    N = env.n_envs
    A = env.size * env.size

    # Storage (CPU)
    obs_me = torch.empty((rollout_len, N), dtype=torch.int64, device="cpu")
    obs_opp = torch.empty((rollout_len, N), dtype=torch.int64, device="cpu")
    actions = torch.empty((rollout_len, N), dtype=torch.int64, device="cpu")
    logprobs = torch.empty((rollout_len, N), dtype=torch.float32, device="cpu")
    values = torch.empty((rollout_len, N), dtype=torch.float32, device="cpu")
    rewards = torch.empty((rollout_len, N), dtype=torch.float32, device="cpu")
    dones = torch.empty((rollout_len, N), dtype=torch.bool, device="cpu")
    legal_mask = torch.empty((rollout_len, N, A), dtype=torch.bool, device="cpu")

    for t in range(rollout_len):
        # Current state from env (numpy uint64); move into CPU torch int64 for storage
        me_np = env.me_bits
        opp_np = env.opp_bits
        done_np = env.done
        mask_np = env.legal_mask  # (N, A) bool

        obs_me[t] = torch.from_numpy(me_np.astype(np.int64))
        obs_opp[t] = torch.from_numpy(opp_np.astype(np.int64))
        dones[t] = torch.from_numpy(done_np.astype(np.bool_))
        legal_mask[t] = torch.from_numpy(mask_np.astype(np.bool_))

        # Policy inference on GPU
        me_t = obs_me[t].to(device)
        opp_t = obs_opp[t].to(device)
        mask_t = legal_mask[t].to(device)

        x = bitboards_to_tensor(me_t, opp_t)  # returns on device
        logits, v = net(x)  # logits (N,A), v (N,)
        probs = masked_softmax(logits, mask_t)

        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()              # (N,)
        lp = dist.log_prob(a)          # (N,)

        # Step env (actions need to be numpy int64)
        sr = env.step(a.detach().to("cpu").numpy().astype(np.int64))

        # Store action + stats
        actions[t] = a.detach().to("cpu")
        logprobs[t] = lp.detach().to("cpu")
        values[t] = v.detach().to("cpu")

        rewards[t] = torch.from_numpy(sr.reward.astype(np.float32))

        # Auto-reset done envs so we always keep N active games
        if np.any(sr.done):
            done_idx = np.where(sr.done)[0]
            # Reset only those envs by zeroing their state (cheap manual reset)
            env.me_bits[done_idx] = np.uint64(0)
            env.opp_bits[done_idx] = np.uint64(0)
            env.done[done_idx] = False
            # Recompute legal mask for all envs (simple; can optimize later)
            occupied = env.me_bits | env.opp_bits
            env.legal_mask = env._compute_legal_mask(occupied)

    # Capture last state for bootstrap value
    last_me = torch.from_numpy(env.me_bits.astype(np.int64))
    last_opp = torch.from_numpy(env.opp_bits.astype(np.int64))
    last_done = torch.from_numpy(env.done.astype(np.bool_))
    last_legal = torch.from_numpy(env.legal_mask.astype(np.bool_))

    return RolloutBatch(
        obs_me=obs_me,
        obs_opp=obs_opp,
        actions=actions,
        logprobs=logprobs,
        values=values,
        rewards=rewards,
        dones=dones,
        legal_mask=legal_mask,
        last_me=last_me,
        last_opp=last_opp,
        last_done=last_done,
        last_legal_mask=last_legal,
    )