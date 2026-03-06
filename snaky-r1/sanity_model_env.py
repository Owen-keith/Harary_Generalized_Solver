import torch

from envs.i4_env_vec import I4EnvVec
from models.resnet_policy_value import PolicyValueNet, bitboards_to_tensor, masked_softmax


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    env = I4EnvVec(n_envs=32, check_legal=True, seed=0)
    me, opp, mask, done = env.reset()

    net = PolicyValueNet().to(device).eval()

    me_t = torch.from_numpy(me).to(device)
    opp_t = torch.from_numpy(opp).to(device)
    mask_t = torch.from_numpy(mask).to(device)

    x = bitboards_to_tensor(me_t, opp_t)
    logits, value = net(x)
    probs = masked_softmax(logits, mask_t)

    print("logits:", tuple(logits.shape))
    print("value:", tuple(value.shape))
    print("probs:", tuple(probs.shape))
    print("probs row sum (env0):", probs[0].sum().item())
    print("illegal probs max (env0):", probs[0][~mask_t[0]].max().item() if (~mask_t[0]).any() else 0.0)


if __name__ == "__main__":
    main()