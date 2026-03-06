import time
import torch

from envs.i4_env_vec import I4EnvVec
from models.resnet_policy_value import PolicyValueNet, bitboards_to_tensor, masked_softmax


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    t1 = time.time()

    env = I4EnvVec(n_envs=32, check_legal=True, seed=0)
    me, opp, mask, done = env.reset()
    t2 = time.time()

    net = PolicyValueNet().to(device).eval()
    t3 = time.time()

    me_t = torch.from_numpy(me).to(device)
    opp_t = torch.from_numpy(opp).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    t4 = time.time()

    x = bitboards_to_tensor(me_t, opp_t)
    t5 = time.time()

    with torch.no_grad():
        logits, value = net(x)
        probs = masked_softmax(logits, mask_t)

        # Force synchronization so timing reflects actual GPU compute
        if device == "cuda":
            torch.cuda.synchronize()
    t6 = time.time()

    print("logits:", tuple(logits.shape))
    print("value:", tuple(value.shape))
    print("probs:", tuple(probs.shape))
    print("probs row sum (env0):", probs[0].sum().item())
    print("illegal probs max (env0):", probs[0][~mask_t[0]].max().item() if (~mask_t[0]).any() else 0.0)

    print("\nTIMINGS (s):")
    print("imports/device checks:", round(t1 - t0, 4))
    print("env init+reset:", round(t2 - t1, 4))
    print("model init+to(device):", round(t3 - t2, 4))
    print("numpy->torch->device:", round(t4 - t3, 4))
    print("bitboards_to_tensor:", round(t5 - t4, 4))
    print("forward+mask:", round(t6 - t5, 4))
    print("TOTAL:", round(t6 - t0, 4))


if __name__ == "__main__":
    main()