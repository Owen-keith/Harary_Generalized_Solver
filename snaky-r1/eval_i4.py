import numpy as np
import torch

from envs.i4_env_vec import I4EnvVec
from models.resnet_policy_value import PolicyValueNet, bitboards_to_tensor, masked_softmax


@torch.no_grad()
def evaluate_selfplay(
    net: torch.nn.Module,
    device: str,
    n_games: int = 2000,
    batch_envs: int = 256,
    greedy: bool = True,
):
    """
    Evaluate net in self-play from empty board.

    Returns dict with:
      - eval/p1_win_rate
      - eval/p2_win_rate
      - eval/draw_rate
      - eval/avg_len

    Robustly tracks which original player (P1/P2) is to move per env using a separate array,
    independent of POV swapping inside the environment.
    """
    assert n_games % batch_envs == 0, "Make n_games divisible by batch_envs for simplicity."
    net.eval()

    total_p1_wins = 0
    total_p2_wins = 0
    total_draws = 0
    total_len = 0

    games_done = 0
    while games_done < n_games:
        env = I4EnvVec(n_envs=batch_envs, check_legal=False, seed=0)
        env.reset()

        finished = np.zeros(batch_envs, dtype=bool)
        # True if original P1 is the player to move in this env
        turn_is_p1 = np.ones(batch_envs, dtype=bool)
        move_count = np.zeros(batch_envs, dtype=np.int32)

        while not np.all(finished):
            alive_idx = np.where(~finished)[0]

            me_np = env.me_bits[alive_idx].astype(np.int64)
            opp_np = env.opp_bits[alive_idx].astype(np.int64)
            mask_np = env.legal_mask[alive_idx].astype(np.bool_)

            me = torch.from_numpy(me_np).to(device)
            opp = torch.from_numpy(opp_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)

            x = bitboards_to_tensor(me, opp)
            logits, _ = net(x)
            probs = masked_softmax(logits, mask)

            if greedy:
                a_alive = torch.argmax(probs, dim=1)
            else:
                dist = torch.distributions.Categorical(probs=probs)
                a_alive = dist.sample()

            a_alive_np = a_alive.to("cpu").numpy().astype(np.int64)

            actions = np.zeros(batch_envs, dtype=np.int64)
            actions[alive_idx] = a_alive_np

            sr = env.step(actions)

            # env.step returns arrays for all envs; only alive_idx were meaningful
            just_done = sr.done & (~finished)
            if np.any(just_done):
                # Winner attribution: if reward == 1, the mover (current turn) won
                won = sr.reward[just_done] == 1.0
                p1_moved = turn_is_p1[just_done]

                # If won and p1_moved => P1 win; if won and not p1_moved => P2 win; else draw
                total_p1_wins += int(np.sum(won & p1_moved))
                total_p2_wins += int(np.sum(won & (~p1_moved)))
                total_draws += int(np.sum(~won))

                finished[just_done] = True

            # Flip turn for envs that are still alive AFTER the move
            still_alive = ~finished
            turn_is_p1[still_alive] = ~turn_is_p1[still_alive]
            move_count[still_alive] += 1

        total_len += int(move_count.sum())
        games_done += batch_envs

    return {
        "eval/p1_win_rate": total_p1_wins / n_games,
        "eval/p2_win_rate": total_p2_wins / n_games,
        "eval/draw_rate": total_draws / n_games,
        "eval/avg_len": total_len / n_games,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = PolicyValueNet().to(device)
    metrics = evaluate_selfplay(net, device=device, n_games=1024, batch_envs=256)
    print(metrics)