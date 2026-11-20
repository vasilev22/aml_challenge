"""
Evaluate the SD sweep variants (sd_sweep.py) against the current coarse stage
(0.2 x ViT-B + 0.8 x trained head, out-of-fold), all on cached maps.

For every variant: PCK alone, and the best blend with the current coarse stage
(raw similarities and per-query z-scored), plus layer combinations (up1 + up2).
The best blend gets a paired McNemar test against the current coarse stage.

Usage
  python experiments/eval_sd_sweep.py --timestep 261
"""

import os
import sys
import glob
import argparse
from math import comb

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_maps import load_maps, predict, target_array, errors  # noqa: E402

CROPS = "s0.25_t0.15_m0.25_rect"
WINDOWS = [("window", 1, 0.03), ("window", 2, 0.03), ("window", 2, 0.05), ("window", 3, 0.05)]


def zs(m):
    f = m.reshape(len(m), -1)
    return ((f - f.mean(1, keepdim=True)) / f.std(1, keepdim=True).clamp_min(1e-6)).view_as(m)


def mcnemar(a, b):
    gain, loss = int((~a & b).sum()), int((a & ~b).sum())
    n = gain + loss
    p = 2 * sum(comb(n, k) for k in range(0, min(gain, loss) + 1)) / 2 ** n if n else 1.0
    return gain, loss, min(p, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", type=int, default=261)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--prefix", default="sdx", help="sdx (SD 2.1) or sdx-sd15")
    args = parser.parse_args()

    q = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    ids, T, S = q.id.tolist(), target_array(q), q.scale.to_numpy()
    sd_files = sorted(glob.glob(f"cache/simmaps/train_{args.prefix}_*_t{args.timestep}_{CROPS}.pt"))
    names = [os.path.basename(f).split(f"_{args.prefix}_")[1].split(f"_t{args.timestep}")[0] for f in sd_files]
    base_files = [f"cache/simmaps/train_dinov2_vitb14_798_fp16_{CROPS}.pt",
                  "cache/simmaps/train_head_conv_h384_o256_nested.pt"]
    z, _, r = load_maps(base_files + sd_files, ids)       # all resampled to the largest grid (96)
    base = (0.25 * z[0] + 1.0 * z[1]) / 1.25
    sd = dict(zip(names, z[2:]))

    def score(m):
        best = (0, None, None)
        for w in WINDOWS:
            e = errors(predict(m, r, w), T, S)
            if (e < .1).mean() > best[0]:
                best = ((e < .1).mean(), w, e < .1)
        return best

    b_pck, _, b_ok = score(base)
    print(f"current coarse stage (ViT-B + head): {b_pck:.4f}\n")
    rows, best_overall = [], (b_pck, "none", b_ok)
    candidates = dict(sd)
    for e in [2, 4, 8]:
        for fl in ["flip", "noflip"]:
            a, b = f"up1_e{e}_{fl}", f"up2_e{e}_{fl}"
            if a in sd and b in sd:
                candidates[f"up1+up2_e{e}_{fl}"] = (sd[a] + sd[b]) / 2
    for name, m in candidates.items():
        alone = score(m)[0]
        best = (0, None)
        for mode, mm, bb in [("raw", m, base), ("z", zs(m), zs(base))]:
            for wsd in [0.1, 0.2, 0.3, 0.4, 0.5]:
                pck, _, ok = score((1 - wsd) * bb + wsd * mm)
                if pck > best[0]:
                    best = (pck, f"{mode} w={wsd}", ok)
        rows.append({"variant": name, "SD alone": alone, "best blend": best[0], "blend": best[1],
                     "gain vs current": best[0] - b_pck})
        if best[0] > best_overall[0]:
            best_overall = (best[0], f"{name} [{best[1]}]", best[2])
    table = pd.DataFrame(rows).sort_values("best blend", ascending=False)
    print(table.round(4).to_string(index=False))
    gain, loss, p = mcnemar(b_ok, best_overall[2])
    print(f"\nBEST: {best_overall[1]} -> {best_overall[0]:.4f} "
          f"(fixed {gain}, broke {loss}, McNemar p = {p:.3f})")
    table.to_csv(f"experiments/results/{args.prefix}_sweep_t{args.timestep}.csv", index=False)


if __name__ == "__main__":
    main()
