"""
Step 3: point-level refinement (a second, per-query zoom).

1. Coarse prediction = best blend of cached maps (same formula as evaluate_maps.py).
2. Per query: a square window around the coarse point in the target
   (side = --frac * size of the target object crop) and a window around the source point
   in the source (side scaled by the source->target crop-size ratio).
3. Features for both windows at --res (flip TTA) -> local similarity map.
4. Writes a map file compatible with evaluate_maps.py:
       zoom    = local similarity map on the window grid
       glob    = coarse score resampled onto the window grid (acts as a prior when blended)
       regions = target windows
   so `evaluate_maps.py --maps <this file>` sweeps the local/coarse blend (alpha) and the window.

Usage
  python experiments/refine_points.py --split train --maps <coarse train map files> \
      --weights 1 --alpha 0.2 --window 2 0.05 --frac 0.35 --res 448
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dino_v2_param_testing import sample_source  # noqa: E402
from dino_v2_geometric import grid_centers  # noqa: E402
from zoom_pipeline import Backbone, extract_many, load_image, DEVICE  # noqa: E402
from evaluate_maps import load_maps, blend, predict  # noqa: E402

MIN_WINDOW_PX = 48


def square_box(center, side, size):
    width, height = size
    sw, sh = min(side, float(width)), min(side, float(height))
    x0 = min(max(center[0] - sw / 2, 0.0), width - sw)
    y0 = min(max(center[1] - sh / 2, 0.0), height - sh)
    return (x0, y0, x0 + sw, y0 + sh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--maps", nargs="+", required=True, help="coarse map files (train_* names)")
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--window", type=float, nargs=2, default=[2, 0.05], help="radius temperature")
    parser.add_argument("--frac", type=float, default=0.35)
    parser.add_argument("--res", type=int, default=448)
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--name", default=None, help="suffix naming the coarse config")
    args = parser.parse_args()

    start = time.time()
    queries = pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
    pairs = {p["pair_id"]: p for p in
             json.load(open(os.path.join(args.data_dir, f"{args.split}_pairs.json")))["pairs"]}
    ids = queries["id"].tolist()

    map_paths = [os.path.join(os.path.dirname(p), os.path.basename(p).replace("train_", f"{args.split}_", 1))
                 for p in args.maps]
    crops = torch.load(map_paths[0])["crops"]
    boxes = json.load(open(os.path.join(args.cache_dir, "boxes", f"{args.split}_{crops}.json")))

    zooms, globs, regions = load_maps(map_paths, ids)
    weights = args.weights or [1.0] * len(map_paths)
    method = ("window", int(args.window[0]), float(args.window[1]))
    score = blend(zooms, globs, weights, args.alpha)                    # [N, g, g] on DEVICE
    coarse = predict(score, regions, method)                            # [N, 2] numpy
    del zooms, globs

    backbone = Backbone(args.backbone)
    g = args.res // backbone.patch
    n = len(queries)
    local_maps = torch.zeros(n, g, g, dtype=torch.float16)
    prior_maps = torch.zeros(n, g, g, dtype=torch.float16)
    windows = torch.zeros(n, 4, dtype=torch.float32)

    for pair_id, group in tqdm(queries.groupby("pair_id", sort=False), desc="refine"):
        meta = pairs[pair_id]
        ss, ts = tuple(meta["source_size"]), tuple(meta["target_size"])
        source_image = load_image(os.path.join(args.data_dir, meta["source_image"]))
        target_image = load_image(os.path.join(args.data_dir, meta["target_image"]))

        b = boxes[pair_id]
        if b["target_box"] is not None:
            sb, tb = b["source_box"], b["target_box"]
            target_size = max(tb[2] - tb[0], tb[3] - tb[1])
            ratio = target_size / max(sb[2] - sb[0], sb[3] - sb[1])
        else:
            target_size = max(ts)
            ratio = max(ts) / max(ss)
        side_t = max(args.frac * target_size, MIN_WINDOW_PX)
        side_s = max(side_t / ratio, MIN_WINDOW_PX)

        rows = group.index.to_numpy()
        items = []
        for r in rows:
            items.append((source_image, square_box((queries.at[r, "source_x"], queries.at[r, "source_y"]), side_s, ss)))
            items.append((target_image, square_box(coarse[r], side_t, ts)))
        features = extract_many(backbone, items, args.res)

        for k, r in enumerate(rows):
            (fs, rs), (ft, rt) = features[2 * k], features[2 * k + 1]
            point = torch.tensor([[queries.at[r, "source_x"] - rs[0], queries.at[r, "source_y"] - rs[1]]],
                                 dtype=torch.float32, device=DEVICE)
            vector = sample_source(fs, point, (rs[2], rs[3]), "bilinear")
            local = (vector @ ft.flatten(1)).view(g, g)

            # coarse score resampled onto the window grid (coords relative to the coarse crop region)
            cx0, cy0, cw, ch = regions[r].tolist()
            centers = grid_centers(g, g, rt)
            grid = torch.stack([(centers[:, 0] - cx0) / cw * 2 - 1, (centers[:, 1] - cy0) / ch * 2 - 1], -1)
            prior = F.grid_sample(score[r][None, None], grid.view(1, g, g, 2), mode="bilinear",
                                  padding_mode="border", align_corners=False)[0, 0]

            local_maps[r] = local.half().cpu()
            prior_maps[r] = prior.half().cpu()
            windows[r] = torch.tensor(rt, dtype=torch.float32)

    name = args.name or f"w{'-'.join(str(w) for w in weights)}_a{args.alpha}_r{method[1]}t{method[2]}"
    out = os.path.join(args.cache_dir, "simmaps",
                       f"{args.split}_refine_{args.backbone}_{args.res}_f{args.frac}_{name}.pt")
    torch.save({"ids": ids, "zoom": local_maps, "glob": prior_maps, "regions": windows,
                "backbone": args.backbone, "res": args.res, "crops": f"refine_f{args.frac}",
                "split": args.split}, out)
    print(f"Saved {out}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
