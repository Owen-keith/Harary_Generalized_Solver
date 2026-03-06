import torch
import torch.nn.functional as F
from dataclasses import dataclass

from models.resnet_policy_value import bitboards_to_tensor, masked_softmax
from rl.rollout import RolloutBatch


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.3
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 1.0

    # Optimization
    lr: float = 1e-3
    epochs: int = 4
    minibatch_size: int = 8192

    # Auxiliary tactics loss
    aux_coef: float = 0.1
    aux_batch_size: int = 2048


def compute_gae(
    rewards: torch.Tensor,     # (T,N)
    dones: torch.Tensor,       # (T,N) bool
    values: torch.Tensor,      # (T,N)
    last_value: torch.Tensor,  # (N,)
    gamma: float,
    lam: float
):
    T, N = rewards.shape
    adv = torch.zeros((T, N), dtype=torch.float32)
    last_adv = torch.zeros((N,), dtype=torch.float32)

    for t in reversed(range(T)):
        not_done = (~dones[t]).to(torch.float32)
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_adv = delta + gamma * lam * not_done * last_adv
        adv[t] = last_adv

    ret = adv + values
    return adv, ret


def masked_cross_entropy(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, neg_inf)
    return F.cross_entropy(masked_logits, target)


def ppo_update(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    device: str,
    cfg: PPOConfig,
    aux_batch=None,  # (me_np, opp_np, mask_np, tgt_np) or None
):
    T, N = batch.rewards.shape
    A = batch.legal_mask.shape[-1]
    total_steps = T * N

    # Bootstrap value for final state
    with torch.no_grad():
        me_last = batch.last_me.to(device)
        opp_last = batch.last_opp.to(device)
        x_last = bitboards_to_tensor(me_last, opp_last)
        _, last_v = net(x_last)  # (N,)

    adv, ret = compute_gae(
        rewards=batch.rewards,
        dones=batch.dones,
        values=batch.values,
        last_value=last_v.detach().to("cpu"),
        gamma=cfg.gamma,
        lam=cfg.gae_lambda
    )

    obs_me = batch.obs_me.reshape(total_steps)
    obs_opp = batch.obs_opp.reshape(total_steps)
    actions = batch.actions.reshape(total_steps)
    old_logp = batch.logprobs.reshape(total_steps)
    returns = ret.reshape(total_steps)
    advantages = adv.reshape(total_steps)
    legal_mask = batch.legal_mask.reshape(total_steps, A)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_me_d = obs_me.to(device)
    obs_opp_d = obs_opp.to(device)
    actions_d = actions.to(device)
    old_logp_d = old_logp.to(device)
    returns_d = returns.to(device)
    advantages_d = advantages.to(device)
    legal_mask_d = legal_mask.to(device)

    idx = torch.arange(total_steps, device=device)

    approx_kl_sum = 0.0
    ent_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    n_mb = 0

    for _ in range(cfg.epochs):
        perm = idx[torch.randperm(total_steps, device=device)]
        for start in range(0, total_steps, cfg.minibatch_size):
            mb_idx = perm[start:start + cfg.minibatch_size]
            if mb_idx.numel() == 0:
                continue

            mb_me = obs_me_d[mb_idx]
            mb_opp = obs_opp_d[mb_idx]
            mb_act = actions_d[mb_idx]
            mb_old_logp = old_logp_d[mb_idx]
            mb_ret = returns_d[mb_idx]
            mb_adv = advantages_d[mb_idx]
            mb_mask = legal_mask_d[mb_idx]

            x = bitboards_to_tensor(mb_me, mb_opp)
            logits, v = net(x)
            probs = masked_softmax(logits, mb_mask)
            dist = torch.distributions.Categorical(probs=probs)

            new_logp = dist.log_prob(mb_act)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - mb_old_logp)

            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            v_loss = F.mse_loss(v, mb_ret)

            loss = policy_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_logp - new_logp).mean().item()
                approx_kl_sum += approx_kl
                ent_sum += entropy.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += v_loss.item()
                n_mb += 1

    metrics = {
        "loss/policy": policy_loss_sum / max(n_mb, 1),
        "loss/value": value_loss_sum / max(n_mb, 1),
        "stats/entropy": ent_sum / max(n_mb, 1),
        "stats/approx_kl": approx_kl_sum / max(n_mb, 1),
        "stats/return_mean": returns.mean().item(),
        "stats/adv_mean": advantages.mean().item(),
    }

    # Optional auxiliary supervised step (tactics)
    if aux_batch is not None:
        me_np, opp_np, mask_np, tgt_np = aux_batch
        if len(me_np) > 0:
            B = len(me_np)
            if B > cfg.aux_batch_size:
                sel = torch.randperm(B)[:cfg.aux_batch_size].cpu().numpy()
                me_np = me_np[sel]
                opp_np = opp_np[sel]
                mask_np = mask_np[sel]
                tgt_np = tgt_np[sel]

            me = torch.from_numpy(me_np).to(device)
            opp = torch.from_numpy(opp_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            tgt = torch.from_numpy(tgt_np).to(device)

            x = bitboards_to_tensor(me, opp)
            logits, _ = net(x)
            aux_loss = masked_cross_entropy(logits, mask, tgt)

            optimizer.zero_grad(set_to_none=True)
            (cfg.aux_coef * aux_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            metrics["loss/aux_ce"] = float(aux_loss.item())
            metrics["stats/aux_batch_size"] = int(len(me_np))
        else:
            metrics["loss/aux_ce"] = 0.0
            metrics["stats/aux_batch_size"] = 0

    return metrics