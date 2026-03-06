import numpy as np

from envs.i4_env_vec import precompute_i4_win_masks


WIN_MASKS = precompute_i4_win_masks(7)


def _is_win(bits: np.uint64) -> bool:
    for m in WIN_MASKS:
        if (bits & m) == m:
            return True
    return False


def legal_moves_from_mask(mask_row: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask_row)


def win_moves_in_1(me_bits: np.uint64, legal_mask_row: np.ndarray) -> list[int]:
    """All legal actions that immediately win for the current player."""
    wins = []
    for a in legal_moves_from_mask(legal_mask_row):
        b = np.uint64(1) << np.uint64(a)
        if _is_win(me_bits | b):
            wins.append(int(a))
    return wins


def block_moves_in_1(me_bits: np.uint64, opp_bits: np.uint64, legal_mask_row: np.ndarray) -> list[int]:
    """
    Moves that block opponent's immediate win (opponent has win-in-1 next turn).
    For I4, blocking means playing in any of opponent's immediate winning squares.
    """
    opp_wins = win_moves_in_1(opp_bits, legal_mask_row)
    return opp_wins


def deterministic_target_for_state(me_bits: np.uint64, opp_bits: np.uint64, legal_mask_row: np.ndarray):
    """
    Returns a deterministic target action:
      - If current player has win-in-1: choose lowest-index winning move
      - Else if opponent has win-in-1: choose lowest-index blocking move
      - Else: None
    """
    wins = win_moves_in_1(me_bits, legal_mask_row)
    if wins:
        return min(wins)

    blocks = block_moves_in_1(me_bits, opp_bits, legal_mask_row)
    if blocks:
        return min(blocks)

    return None


def generate_tactic_batch_from_env(env, n_samples: int, rng: np.random.Generator):
    """
    Sample tactic states from the current env batch.

    Returns:
      me_bits int64 shape (B,)
      opp_bits int64 shape (B,)
      legal_mask bool shape (B,49)
      target_actions int64 shape (B,)
    May return None if no tactic states found (unlikely).
    """
    N = env.n_envs
    idxs = rng.integers(0, N, size=n_samples, endpoint=False)

    me_list = []
    opp_list = []
    mask_list = []
    tgt_list = []

    for i in idxs:
        if env.done[i]:
            continue

        me = env.me_bits[i]
        opp = env.opp_bits[i]
        mask = env.legal_mask[i]

        tgt = deterministic_target_for_state(me, opp, mask)
        if tgt is None:
            continue

        me_list.append(np.int64(me))
        opp_list.append(np.int64(opp))
        mask_list.append(mask.astype(np.bool_))
        tgt_list.append(np.int64(tgt))

    if not me_list:
        return None

    return (
        np.array(me_list, dtype=np.int64),
        np.array(opp_list, dtype=np.int64),
        np.stack(mask_list, axis=0),
        np.array(tgt_list, dtype=np.int64),
    )