import numpy as np
from dataclasses import dataclass

# 7x7 board => 49 squares. We map (r, c) -> idx = r*7 + c, with idx in [0,48].
# Bit idx corresponds to (1 << idx) in a uint64.

def rc_to_idx(r: int, c: int, size: int = 7) -> int:
    return r * size + c


def precompute_i4_win_masks(size: int = 7) -> np.ndarray:
    """
    Returns a uint64 array of shape (num_masks,) where each entry is a bitmask
    for a horizontal or vertical run of length 4.
    For size=7: total = 56.
    """
    masks = []
    L = 4

    # Horizontal
    for r in range(size):
        for c0 in range(size - L + 1):
            m = np.uint64(0)
            for dc in range(L):
                idx = rc_to_idx(r, c0 + dc, size)
                m |= (np.uint64(1) << np.uint64(idx))
            masks.append(m)

    # Vertical
    for c in range(size):
        for r0 in range(size - L + 1):
            m = np.uint64(0)
            for dr in range(L):
                idx = rc_to_idx(r0 + dr, c, size)
                m |= (np.uint64(1) << np.uint64(idx))
            masks.append(m)

    return np.array(masks, dtype=np.uint64)


@dataclass
class StepResult:
    me_bits: np.ndarray      # uint64, shape (n_envs,)
    opp_bits: np.ndarray     # uint64, shape (n_envs,)
    reward: np.ndarray       # float32, shape (n_envs,)
    done: np.ndarray         # bool, shape (n_envs,)
    legal_mask: np.ndarray   # bool, shape (n_envs, 49)


class I4EnvVec:
    """
    Vectorized 7x7 maker-maker environment
    goal = I-tetromino (4 in a row) horizontally or vertically.

    State stored as two bitboards from POV of player-to-move:
      - me_bits: current player stones
      - opp_bits: opponent stones
    After each nonterminal move, swap (me_bits, opp_bits).
    """

    def __init__(self, n_envs: int, size: int = 7, seed: int | None = None, check_legal: bool = False):
        assert size == 7, "This implementation currently assumes 7x7 (49 squares) for uint64 bitboard."
        self.size = size
        self.n_envs = int(n_envs)
        self.check_legal = bool(check_legal)

        self.rng = np.random.default_rng(seed)

        self.win_masks = precompute_i4_win_masks(size=self.size)  # (56,)
        self.full_mask = (np.uint64(1) << np.uint64(self.size * self.size)) - np.uint64(1)

        self.me_bits = np.zeros(self.n_envs, dtype=np.uint64)
        self.opp_bits = np.zeros(self.n_envs, dtype=np.uint64)
        self.done = np.zeros(self.n_envs, dtype=np.bool_)

        self.legal_mask = np.ones((self.n_envs, self.size * self.size), dtype=np.bool_)

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.me_bits.fill(np.uint64(0))
        self.opp_bits.fill(np.uint64(0))
        self.done.fill(False)
        self.legal_mask[:] = True
        return self.me_bits.copy(), self.opp_bits.copy(), self.legal_mask.copy(), self.done.copy()

    def _compute_legal_mask(self, occupied_bits: np.ndarray) -> np.ndarray:
        """
        occupied_bits: uint64 array shape (B,)
        Returns bool array shape (B, 49) with True for empty squares.
        """
        B = occupied_bits.shape[0]
        lm = np.empty((B, self.size * self.size), dtype=np.bool_)
        for idx in range(self.size * self.size):
            bit = (np.uint64(1) << np.uint64(idx))
            lm[:, idx] = (occupied_bits & bit) == 0
        return lm

    def _is_win(self, bits: np.ndarray) -> np.ndarray:
        """
        bits: uint64 array shape (B,)
        Returns bool array shape (B,) indicating if that player has 4-in-a-row.
        """
        B = bits.shape[0]
        w = np.zeros(B, dtype=np.bool_)
        for m in self.win_masks:
            w |= (bits & m) == m
        return w

    def step(self, actions: np.ndarray) -> StepResult:
        """
        actions: int64 array shape (n_envs,), each in [0,48].
        For envs already done, action is ignored and state stays done.
        """
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.n_envs,)

        reward = np.zeros(self.n_envs, dtype=np.float32)

        alive = ~self.done
        if not np.any(alive):
            return StepResult(
                me_bits=self.me_bits.copy(),
                opp_bits=self.opp_bits.copy(),
                reward=reward,
                done=self.done.copy(),
                legal_mask=self.legal_mask.copy(),
            )

        occupied = self.me_bits | self.opp_bits

        if self.check_legal:
            for i in np.where(alive)[0]:
                a = int(actions[i])
                if a < 0 or a >= self.size * self.size:
                    raise ValueError(f"Illegal action {a} out of range in env {i}")
                bit = (np.uint64(1) << np.uint64(a))
                if (occupied[i] & bit) != 0:
                    raise ValueError(f"Illegal action {a} on occupied square in env {i}")

        # Apply moves to alive envs
        idxs = np.where(alive)[0]
        for i in idxs:
            a = int(actions[i])
            self.me_bits[i] |= (np.uint64(1) << np.uint64(a))

        # Compute wins for ALL envs (avoids shape mismatch on slices)
        mover_win_all = self._is_win(self.me_bits)
        just_won = alive & mover_win_all

        occupied_after = self.me_bits | self.opp_bits
        board_full = (occupied_after & self.full_mask) == self.full_mask
        draw = alive & board_full & (~just_won)

        just_done = just_won | draw
        self.done[just_done] = True

        reward[just_won] = 1.0
        reward[draw] = 0.0

        # Swap perspective for continuing envs
        cont = alive & (~just_done)
        if np.any(cont):
            me_new = self.opp_bits[cont].copy()
            opp_new = self.me_bits[cont].copy()
            self.me_bits[cont] = me_new
            self.opp_bits[cont] = opp_new

        # Update legal mask for all envs
        occupied_final = self.me_bits | self.opp_bits
        self.legal_mask = self._compute_legal_mask(occupied_final)

        return StepResult(
            me_bits=self.me_bits.copy(),
            opp_bits=self.opp_bits.copy(),
            reward=reward,
            done=self.done.copy(),
            legal_mask=self.legal_mask.copy(),
        )

    def sample_random_actions(self) -> np.ndarray:
        actions = np.zeros(self.n_envs, dtype=np.int64)
        for i in range(self.n_envs):
            if self.done[i]:
                actions[i] = 0
                continue
            legal = np.flatnonzero(self.legal_mask[i])
            actions[i] = int(self.rng.choice(legal))
        return actions