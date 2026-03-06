from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bitboards_to_tensor(me_bits: torch.Tensor,
                        opp_bits: torch.Tensor,
                        board_size: int = 7) -> torch.Tensor:
    """
    Convert bitboards to a (B, 2, 7, 7) float tensor with 0/1 planes.

    me_bits, opp_bits: torch tensors of shape (B,) dtype torch.uint64 (or int64 ok)
    Returns:
        x: float32 tensor shape (B, 2, board_size, board_size)
    """
    assert board_size == 7, "This helper currently assumes 7x7 -> 49 bits."
    B = me_bits.shape[0]
    device = me_bits.device

    # Create bit positions [0..48]
    idx = torch.arange(board_size * board_size, device=device, dtype=torch.int64)  # (49,)
    # Expand to (B,49)
    idx = idx.unsqueeze(0).expand(B, -1)

    # Ensure unsigned-ish shifting works: cast to uint64 for bit ops
    me_u = me_bits.to(torch.uint64).unsqueeze(1)   # (B,1)
    opp_u = opp_bits.to(torch.uint64).unsqueeze(1) # (B,1)

    # (B,49) boolean occupancy by testing each bit
    me_plane = ((me_u >> idx.to(torch.uint64)) & 1).to(torch.float32)
    opp_plane = ((opp_u >> idx.to(torch.uint64)) & 1).to(torch.float32)

    # Reshape to (B,7,7) and stack channels -> (B,2,7,7)
    me_plane = me_plane.view(B, board_size, board_size)
    opp_plane = opp_plane.view(B, board_size, board_size)

    x = torch.stack([me_plane, opp_plane], dim=1)
    return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = F.relu(out, inplace=True)
        return out


class PolicyValueNet(nn.Module):
    """
    Small ResNet-style CNN for 7x7 board.
    Input: (B, 2, 7, 7) float {0,1}
    Output:
      - logits: (B, 49)
      - value: (B,) in [-1, 1] via tanh
    """

    def __init__(self, board_size: int = 7, in_channels: int = 2,
                 channels: int = 64, n_blocks: int = 6):
        super().__init__()
        assert board_size == 7, "Current code assumes 7x7 (49 actions)."
        self.board_size = board_size
        self.n_actions = board_size * board_size

        # Stem
        self.conv0 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(channels)

        # Residual trunk
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        # Policy head
        self.p_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board_size * board_size, self.n_actions)

        # Value head
        self.v_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * board_size * board_size, 128)
        self.v_fc2 = nn.Linear(128, 1)

        # Init
        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming init for conv/linear; BN default is fine
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,2,7,7)
        returns:
          logits: (B,49)
          value: (B,) tanh bounded
        """
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out, inplace=True)

        out = self.blocks(out)

        # Policy
        p = self.p_conv(out)
        p = self.p_bn(p)
        p = F.relu(p, inplace=True)
        p = p.flatten(1)
        logits = self.p_fc(p)

        # Value
        v = self.v_conv(out)
        v = self.v_bn(v)
        v = F.relu(v, inplace=True)
        v = v.flatten(1)
        v = F.relu(self.v_fc1(v), inplace=True)
        value = torch.tanh(self.v_fc2(v)).squeeze(1)

        return logits, value


@torch.no_grad()
def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Utility: convert logits -> probs with a boolean mask.
    logits: (B, A)
    legal_mask: (B, A) bool, True where legal
    Returns probs (B,A) with zero on illegal actions.
    """
    # Set illegal to very negative
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~legal_mask, neg_inf)
    probs = torch.softmax(masked_logits, dim=dim)
    # Softmax should already give ~0 on illegal; explicitly zero for safety
    probs = probs * legal_mask.to(probs.dtype)
    # Renormalize (in case all legal were masked somehow)
    probs = probs / (probs.sum(dim=dim, keepdim=True) + 1e-8)
    return probs