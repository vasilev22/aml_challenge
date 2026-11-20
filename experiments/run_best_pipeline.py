"""
Reproduces submissions/fused_vote_refineBL_recipe.csv (best Kaggle public score, 0.594; local PCK 0.6108).

Pipeline (every step skips itself if its output already exists in cache/):

  1. Zoom, ViT-B       zoom_pipeline.py: affine crop boxes + DINOv2 ViT-B/14 similarity maps (798px crops)
  2. Zoom, ViT-L       same crops, DINOv2 ViT-L/14 (HF weights), also saves the crop features
  3. SD features       sd_sweep.py: Stable Diffusion 2.1 decoder block 1, t=101, 4 noise draws, flip-averaged
  4. Head              train_head.py --nested: small conv head on ViT-L features (5-fold, image-disjoint)
  5. Fused head        same, on ViT-L + SD features concatenated
  6. Coarse vote       0.25 x ViT-B + 0.75 x head + 0.5 x fused head  ->  coarse point per query
     Refinement        refine_points.py: 40% window around the coarse point, re-matched at 448px
                       with ViT-B and ViT-L
  7. Final             0.75 x ViT-B window + 0.25 x ViT-L window, coarse prior alpha 0.6,
                       windowed soft-argmax (r=2, t=0.03)  ->  submission CSV

Train maps from the heads are out-of-fold, so the local PCK printed at the end is an honest estimate.
Test maps from the heads average the 5 fold models. Head training on GPU is not bit-exact, so a
from-scratch rerun lands close to, but not exactly on, the cached result.

Usage
  python experiments/run_best_pipeline.py            # needs a CUDA GPU for steps 1-6
  python experiments/run_best_pipeline.py --out submissions/my_rerun.csv
"""

import os
import sys
import json
import argparse
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_maps import load_maps, blend, predict, target_array, errors  # noqa: E402

C = "s0.25_t0.15_m0.25_rect"           # crop setting: source margin 0.25, target margin 0.15, min 25%
SM, FE = "cache/simmaps", "cache/features"
PY = sys.executable
SPLITS = ["train", "test"]

VITB_MAPS = f"{SM}/train_dinov2_vitb14_798_fp16_{C}.pt"
VITL_FEATS = f"{FE}/train_dinov2_vitl14_hf_798_fp16_{C}.pt"
SD_FEATS = f"{FE}/train_sdx_up1_e4_flip_t101_{C}.pt"
HEAD_MAPS = f"{SM}/train_head_conv_h384_o256_nested.pt"
FUSED_MAPS = f"{SM}/train_head_conv_h384_o256_nested_fused.pt"
COARSE = {VITB_MAPS: 0.25, HEAD_MAPS: 0.75, FUSED_MAPS: 0.5}
REFINE = {"dinov2_vitb14": 0.75, "dinov2_vitl14_hf": 0.25}
REFINE_MAPS = [f"{SM}/train_refine_{b}_448_f0.4_coarse_BHF.pt" for b in REFINE]
ALPHA, METHOD = 0.6, ("window", 2, 0.03)


def step(name, outputs, cmd):
    if all(os.path.exists(o) for o in outputs):
        print(f"[skip] {name} (cached)")
        return
    print(f"[run ] {name}\n       {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, check=True)


def test(path):
    return path.replace("/train_", "/test_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submissions/fused_vote_refineBL_recipe.csv")
    args = parser.parse_args()

    for s in SPLITS:
        step(f"1. zoom ViT-B ({s})", [VITB_MAPS.replace("/train_", f"/{s}_")],
             [PY, "experiments/zoom_pipeline.py", "--split", s, "--backbone", "dinov2_vitb14"])
    for s in SPLITS:
        step(f"2. zoom ViT-L ({s})", [VITL_FEATS.replace("/train_", f"/{s}_")],
             [PY, "experiments/zoom_pipeline.py", "--split", s, "--backbone", "dinov2_vitl14_hf", "--save_features"])
    for s in SPLITS:
        step(f"3. SD 2.1 features ({s})", [SD_FEATS.replace("/train_", f"/{s}_")],
             [PY, "experiments/sd_sweep.py", "--split", s, "--sd", "sd21", "--timestep", "101", "--samples", "4",
              "--only", "up1_e4_flip", "--save_features", "up1_e4_flip"])
    step("4. head on ViT-L", [HEAD_MAPS, test(HEAD_MAPS)],
         [PY, "experiments/train_head.py", "--nested", "--features", VITL_FEATS])
    step("5. fused head on ViT-L + SD", [FUSED_MAPS, test(FUSED_MAPS)],
         [PY, "experiments/train_head.py", "--nested", "--features", VITL_FEATS, "--extra_features", SD_FEATS])
    for s in SPLITS:
        for backbone, out in zip(REFINE, REFINE_MAPS):
            step(f"6. refine with {backbone} ({s})", [out.replace("/train_", f"/{s}_")],
                 [PY, "experiments/refine_points.py", "--split", s, "--backbone", backbone,
                  "--maps", *COARSE, "--weights", *map(str, COARSE.values()), "--alpha", "0.0",
                  "--window", "2", "0.03", "--frac", "0.4", "--res", "448", "--name", "coarse_BHF"])

    print("[run ] 7. final blend")
    weights = list(REFINE.values())
    q = pd.read_csv("data/train.csv")
    z, g, r = load_maps(REFINE_MAPS, q.id.tolist())
    e = errors(predict(blend(z, g, weights, ALPHA), r, METHOD), target_array(q), q.scale.to_numpy())
    print(f"       local PCK@0.10 (train, out-of-fold) = {np.mean(e < .1):.4f}")

    t = pd.read_csv("data/test.csv")
    z, g, r = load_maps([test(p) for p in REFINE_MAPS], t.id.tolist())
    P = predict(blend(z, g, weights, ALPHA), r, METHOD)
    pd.DataFrame({"id": t.id, "prediction": [json.dumps([round(float(x), 2), round(float(y), 2)]) for x, y in P]}) \
        .to_csv(args.out, index=False)
    print(f"       saved {args.out} ({len(P)} rows)")


if __name__ == "__main__":
    main()
