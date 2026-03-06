import argparse
import numpy as np
import torch

from envs.i4_env_vec import I4EnvVec
from models.resnet_policy_value import PolicyValueNet, bitboards_to_tensor, masked_softmax


def render_board(me_bits: np.uint64, opp_bits: np.uint64) -> str:
    """
    Render from POV of player-to-move:
      'X' = current player's stones (me_bits)
      'O' = opponent stones (opp_bits)
      '.' = empty
    """
    size = 7
    out = []
    out.append("   " + " ".join(str(c) for c in range(size)))
    for r in range(size):
        row = []
        for c in range(size):
            idx = r * size + c
            bit = np.uint64(1) << np.uint64(idx)
            if (me_bits & bit) != 0:
                row.append("X")
            elif (opp_bits & bit) != 0:
                row.append("O")
            else:
                row.append(".")
        out.append(f"{r}  " + " ".join(row))
    out.append("")
    out.append("Legend (POV of player to move): X=to-move, O=other")
    return "\n".join(out)


def idx_to_rc(idx: int) -> tuple[int, int]:
    return idx // 7, idx % 7


@torch.no_grad()
def agent_move(
    net: torch.nn.Module,
    device: str,
    me_bits: np.uint64,
    opp_bits: np.uint64,
    legal_mask: np.ndarray,
    temperature: float = 0.0,
    topk: int = 5
) -> int:
    """
    Choose an action for the current player.
    temperature=0 -> greedy
    temperature>0 -> sample from softmax(logits/temperature)
    """
    me = torch.tensor([int(me_bits)], dtype=torch.int64, device=device)
    opp = torch.tensor([int(opp_bits)], dtype=torch.int64, device=device)
    mask = torch.from_numpy(legal_mask.astype(np.bool_)).unsqueeze(0).to(device)

    x = bitboards_to_tensor(me, opp)
    logits, _ = net(x)

    if temperature and temperature > 1e-8:
        logits = logits / temperature

    probs = masked_softmax(logits, mask)[0]  # (49,)

    # Top-k suggestions
    legal_count = int(legal_mask.sum())
    if legal_count > 0:
        k = min(topk, legal_count)
        pvals, aidx = torch.topk(probs, k=k)
        print("Agent top moves:")
        for p, a in zip(pvals.tolist(), aidx.tolist()):
            r, c = idx_to_rc(a)
            print(f"  idx {a:2d} -> ({r},{c})  p={p:.3f}")
        print("")

    if temperature and temperature > 1e-8:
        dist = torch.distributions.Categorical(probs=probs)
        a = int(dist.sample().item())
    else:
        a = int(torch.argmax(probs).item())

    return a


def parse_human_move(legal_mask: np.ndarray) -> int:
    while True:
        s = input("Your move as 'r c' or 'idx': ").strip()
        if not s:
            continue
        parts = s.split()
        try:
            if len(parts) == 1:
                idx = int(parts[0])
                if 0 <= idx < 49 and legal_mask[idx]:
                    return idx
                print("Illegal idx. Must be 0..48 and empty.")
            elif len(parts) == 2:
                r = int(parts[0])
                c = int(parts[1])
                if 0 <= r < 7 and 0 <= c < 7:
                    idx = r * 7 + c
                    if legal_mask[idx]:
                        return idx
                print("Illegal (r,c). Must be within board and empty.")
            else:
                print("Enter either one number (idx) or two numbers (r c).")
        except ValueError:
            print("Could not parse. Try again.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--human", choices=["p1", "p2"], default="p1", help="Play as original P1 or P2")
    ap.add_argument("--temperature", type=float, default=0.0, help="0=greedy, >0=sampling")
    ap.add_argument("--topk", type=int, default=5, help="Show top-k agent suggestions")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = PolicyValueNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    env = I4EnvVec(n_envs=1, check_legal=True, seed=0)
    env.reset()

    # Track whose turn it is in original labeling (P1 starts)
    turn_is_p1 = True
    human_is_p1 = (args.human == "p1")

    print("")
    print("Game start. Board is rendered from POV of player to move each turn.")
    print("So your stones appear as X when it's your turn, but as O when it's not your turn.")
    print("")

    while True:
        me_bits = env.me_bits[0]
        opp_bits = env.opp_bits[0]
        legal = env.legal_mask[0]

        human_to_move = (turn_is_p1 == human_is_p1)

        print(render_board(me_bits, opp_bits))
        print("To move:", "HUMAN" if human_to_move else "AGENT", f"({'P1' if turn_is_p1 else 'P2'})")

        if human_to_move:
            a = parse_human_move(legal)
        else:
            a = agent_move(
                net=net,
                device=device,
                me_bits=me_bits,
                opp_bits=opp_bits,
                legal_mask=legal,
                temperature=args.temperature,
                topk=args.topk
            )
            r, c = idx_to_rc(a)
            print(f"Agent plays: idx {a} -> ({r},{c})\n")

        sr = env.step(np.array([a], dtype=np.int64))

        if sr.done[0]:
            # sr.reward[0] == 1 => mover won; else draw
            if sr.reward[0] == 1.0:
                winner = "P1" if turn_is_p1 else "P2"
                human_won = (winner == ("P1" if human_is_p1 else "P2"))
                print(render_board(env.me_bits[0], env.opp_bits[0]))
                print(f"GAME OVER: {winner} wins. {'(You won!)' if human_won else '(Agent won.)'}")
            else:
                print(render_board(env.me_bits[0], env.opp_bits[0]))
                print("GAME OVER: Draw.")
            break

        # Next player's turn (original labeling)
        turn_is_p1 = not turn_is_p1


if __name__ == "__main__":
    main()