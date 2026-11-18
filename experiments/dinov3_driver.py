"""
Unattended DINOv3 step: choose the coarse-stage voters on honest out-of-fold maps, then apply
the Kaggle-proven refinement recipe and write a submission.

1. Candidate coarse votes (weights grid-searched on train, out-of-fold maps):
     base      : ViT-B raw + DINOv2 head + fused head      (current best, Kaggle 0.594)
     +v3L raw  / +v3L head / +v3L raw+head / +v3B raw
2. Best candidate vs base: paired McNemar test.
3. If a DINOv3 candidate wins: refine_points.py (ViT-B and ViT-L raw windows, train + test) on it.
4. Refinement recipe fixed to the one that scored 0.589 / 0.594 on Kaggle:
     0.75 x ViT-B windows + 0.25 x ViT-L windows, coarse prior 0.6, window r2 t0.03
5. Writes submissions/dinov3_vote_refineBL_recipe.csv and experiments/results/dinov3_summary.txt
"""

import os
import sys
import json
import itertools
import subprocess
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_maps import load_maps, blend, predict, target_array, errors  # noqa: E402

C = "s0.25_t0.15_m0.25_rect"
SM = "cache/simmaps"
FILES = {
    "vitB": f"{SM}/train_dinov2_vitb14_798_fp16_{C}.pt",
    "head": f"{SM}/train_head_conv_h384_o256_nested.pt",
    "fused": f"{SM}/train_head_conv_h384_o256_nested_fused.pt",
    "v3L": f"{SM}/train_dinov3_vitl16_800_fp16_{C}.pt",
    "v3Lhead": f"{SM}/train_head_v3L_nested.pt",
    "v3B": f"{SM}/train_dinov3_vitb16_800_fp16_{C}.pt",
}
BASE_W = {"vitB": 0.25, "head": 0.75, "fused": 0.5}
CANDIDATES = {
    "base": ["vitB", "head", "fused"],
    "+v3L raw": ["vitB", "head", "fused", "v3L"],
    "+v3L head": ["vitB", "head", "fused", "v3Lhead"],
    "+v3L raw+head": ["vitB", "head", "fused", "v3L", "v3Lhead"],
    "+v3B raw": ["vitB", "head", "fused", "v3B"],
}
STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]
COARSE_WINDOW = ("window", 2, 0.03)
RECIPE = dict(weights=[0.75, 0.25], alpha=0.6, method=("window", 2, 0.03))
OUT = "experiments/results/dinov3_summary.txt"


def log(msg):
    print(msg, flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def mcnemar(a, b):
    gain, loss = int((~a & b).sum()), int((a & ~b).sum())
    n = gain + loss
    p = 2 * sum(comb(n, k) for k in range(0, min(gain, loss) + 1)) / 2 ** n if n else 1.0
    return gain, loss, min(p, 1.0)


def main():
    open(OUT, "w").close()
    q = pd.read_csv("data/train.csv")
    ids, T, S = q.id.tolist(), target_array(q), q.scale.to_numpy()
    available = {k: v for k, v in FILES.items() if os.path.exists(v)}
    log(f"available maps: {sorted(available)}")

    results = {}
    for name, members in CANDIDATES.items():
        if not all(m in available for m in members):
            log(f"skip {name}: missing maps")
            continue
        z, g, r = load_maps([available[m] for m in members], ids)
        best = (-1, None, None)
        if name == "base":
            grid = [tuple(BASE_W[m] for m in members)]
        else:   # base voters keep their proportions; new voters get a weight from STEPS
            extra = len(members) - 3
            grid = [tuple(BASE_W[m] for m in members[:3]) + w for w in itertools.product(STEPS, repeat=extra)]
        for w in grid:
            if sum(w) == 0:
                continue
            ok = errors(predict(blend(z, g, list(w), 0.0), r, COARSE_WINDOW), T, S) < .1
            if ok.mean() > best[0]:
                best = (ok.mean(), w, ok)
        results[name] = (best[0], dict(zip(members, best[1])), best[2])
        log(f"{name:16s} coarse PCK {best[0]:.4f}  weights {dict(zip(members, best[1]))}")

    base_ok = results["base"][2]
    winner = max(results, key=lambda k: results[k][0])
    gain, loss, p = mcnemar(base_ok, results[winner][2])
    log(f"\nWINNER: {winner}  {results[winner][0]:.4f} vs base {results['base'][0]:.4f}  "
        f"(fixed {gain}, broke {loss}, McNemar p = {p:.3f})")

    weights = results[winner][1]
    members = list(weights)
    name = "coarse_BHF" if winner == "base" else "coarse_V3"
    if winner != "base":
        log("running refinement windows on the new coarse stage ...")
        for split in ["train", "test"]:
            for backbone in ["dinov2_vitb14", "dinov2_vitl14_hf"]:
                cmd = [sys.executable, "experiments/refine_points.py", "--split", split, "--backbone", backbone,
                       "--maps", *[available[m] for m in members],
                       "--weights", *[str(weights[m]) for m in members],
                       "--alpha", "0.0", "--window", "2", "0.03", "--frac", "0.4", "--res", "448", "--name", name]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    refine = [f"{SM}/train_refine_dinov2_vitb14_448_f0.4_{name}.pt", f"{SM}/train_refine_dinov2_vitl14_hf_448_f0.4_{name}.pt"]
    z, g, r = load_maps(refine, ids)
    e = errors(predict(blend(z, g, RECIPE["weights"], RECIPE["alpha"]), r, RECIPE["method"]), T, S)
    log(f"final (coarse {winner} + Kaggle refinement recipe): local PCK {np.mean(e < .1):.4f}  "
        f"(previous best file fused_vote_refineBL_recipe: local 0.6108, Kaggle 0.594)")

    t = pd.read_csv("data/test.csv")
    tz, tg, tr = load_maps([p.replace("train_", "test_") for p in refine], t.id.tolist())
    P = predict(blend(tz, tg, RECIPE["weights"], RECIPE["alpha"]), tr, RECIPE["method"])
    out = "submissions/dinov3_vote_refineBL_recipe.csv"
    pd.DataFrame({"id": t.id, "prediction": [json.dumps([round(float(x), 2), round(float(y), 2)]) for x, y in P]}).to_csv(out, index=False)
    log(f"Saved {out} ({len(P)} rows)")


if __name__ == "__main__":
    main()
