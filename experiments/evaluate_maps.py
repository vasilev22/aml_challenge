"""
Evaluate / blend cached similarity maps from zoom_pipeline.py (no backbone, seconds per sweep).

    score = sum_b w_b * [ (1 - alpha) * zoom_b + alpha * glob_b ] / sum_b w_b
    prediction = window soft-argmax on score, mapped back through the target-crop region

All map files must use the same crops (same boxes); maps with different grid sizes
(e.g. DINOv2 57x57 vs DINOv3 50x50) are resampled to the largest grid.

Usage
  python experiments/evaluate_maps.py --maps cache/simmaps/train_dinov2_vitb14_798_fp16_s0.25_t0.15_m0.25_rect.pt
  python experiments/evaluate_maps.py --maps A.pt B.pt --tag v2+v3 --submit
         (--submit: apply the best train config to the matching test_*.pt files)
"""

import os
import sys
import json
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dino_v2_param_testing import match  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5]
WINDOWS = [
    ("window", 1, 0.03), ("window", 1, 0.05),
    ("window", 2, 0.03), ("window", 2, 0.05), ("window", 2, 0.08),
    ("window", 3, 0.05),
]
WEIGHT_STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_maps(paths, ids):
    """Load map files, reorder to `ids`, resample to a common grid. Returns zoom, glob lists + regions."""
    data = [torch.load(p) for p in paths]
    size = max(d["zoom"].shape[-1] for d in data)
    zooms, globs, regions = [], [], None
    for d in data:
        index = {q: i for i, q in enumerate(d["ids"])}
        order = torch.tensor([index[q] for q in ids])
        z, g = d["zoom"][order].float(), d["glob"][order].float()
        if z.shape[-1] != size:
            z = F.interpolate(z[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
            g = F.interpolate(g[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
        zooms.append(z.to(DEVICE))
        globs.append(g.to(DEVICE))
        r = d["regions"][order]
        if regions is None:
            regions = r
        elif not torch.allclose(regions, r):
            raise ValueError("Map files use different crops; blend only files with identical boxes.")
    return zooms, globs, regions.to(DEVICE)


def blend(zooms, globs, weights, alpha):
    total = sum(weights)
    return sum(w * ((1 - alpha) * z + alpha * g) for w, z, g in zip(weights, zooms, globs)) / total


def predict(score, regions, method):
    n, h, w = score.shape
    normalized = match(score.reshape(n, -1), h, w, (1.0, 1.0), method)   # [N, 2] in [0, 1]
    return (regions[:, :2] + normalized * regions[:, 2:]).cpu().numpy()


def target_array(queries):
    targets = [np.asarray(json.loads(s)) for s in queries["target_points"]]
    k = max(len(t) for t in targets)
    arr = np.full((len(targets), k, 2), np.nan)
    for i, t in enumerate(targets):
        arr[i, :len(t)] = t
    return arr


def errors(predictions, targets, scale):
    d = np.sqrt(((targets - predictions[:, None]) ** 2).sum(-1))
    return np.nanmin(d, axis=1) / scale


def weight_grid(k):
    if k == 1:
        return [(1.0,)]
    seen, grid = set(), []
    for w in itertools.product(WEIGHT_STEPS, repeat=k):
        if sum(w) == 0:
            continue
        key = tuple(round(x / sum(w), 4) for x in w)
        if key not in seen:
            seen.add(key)
            grid.append(w)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", nargs="+", required=True, help="train map files")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="experiments/results")
    parser.add_argument("--submission_dir", default="submissions")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    args = parser.parse_args()
    tag = args.tag or "+".join(os.path.basename(p).replace("train_", "").replace(".pt", "") for p in args.maps)

    queries = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    ids = queries["id"].tolist()
    targets, scale = target_array(queries), queries["scale"].to_numpy()
    zooms, globs, regions = load_maps(args.maps, ids)

    rows, best = [], None
    for weights in weight_grid(len(args.maps)):
        for alpha in args.alphas:
            score = blend(zooms, globs, weights, alpha)
            for method in WINDOWS:
                e = errors(predict(score, regions, method), targets, scale)
                row = {"weights": weights, "alpha": alpha, "window": f"r{method[1]}_t{method[2]}",
                       "pck@0.10": float((e < 0.10).mean()), "pck@0.05": float((e < 0.05).mean())}
                rows.append(row)
                if best is None or row["pck@0.10"] > best[0]["pck@0.10"]:
                    best = (row, weights, alpha, method, e)

    table = pd.DataFrame(rows).sort_values("pck@0.10", ascending=False)
    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(os.path.join(args.output_dir, f"maps_{tag}.csv"), index=False)
    print(table.head(12).to_string(index=False, float_format="%.4f"))

    row, weights, alpha, method, e = best
    print(f"\nBEST {tag}: weights={weights} alpha={alpha} window={method[1:]} "
          f"PCK@0.10={row['pck@0.10']:.4f} PCK@0.05={row['pck@0.05']:.4f}")
    rel = scale / queries["pair_id"].map(
        {p["pair_id"]: max(p["target_size"]) for p in
         json.load(open(os.path.join(args.data_dir, "train_pairs.json")))["pairs"]}
    ).to_numpy()
    bins = pd.cut(rel, [0, .2, .4, .7, 10])
    print(pd.Series(e < 0.10).groupby(bins, observed=True).mean().round(3).to_string())

    score = blend(zooms, globs, weights, alpha)
    pred = predict(score, regions, method)
    pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "error": e}).to_csv(
        os.path.join(args.output_dir, f"maps_{tag}_best_per_query.csv"), index=False
    )

    if args.submit:
        test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        test_paths = [os.path.join(os.path.dirname(p), os.path.basename(p).replace("train_", "test_", 1))
                      for p in args.maps]
        tz, tg, tr = load_maps(test_paths, test["id"].tolist())
        tp = predict(blend(tz, tg, weights, alpha), tr, method)
        os.makedirs(args.submission_dir, exist_ok=True)
        path = os.path.join(args.submission_dir, f"{tag}.csv")
        pd.DataFrame({
            "id": test["id"],
            "prediction": [json.dumps([round(float(x), 2), round(float(y), 2)]) for x, y in tp],
        }).to_csv(path, index=False)
        print("Saved submission", path)


if __name__ == "__main__":
    main()
