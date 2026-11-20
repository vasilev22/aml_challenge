"""
Generate Kaggle submissions for the 5 best DINOv2 configurations found by
dino_v2_param_testing.py (full sweep on all 2,495 train queries, PCK@0.10).

Rank  res  resize       flip  source    matching        train PCK@0.10
  1   798  square       yes   bilinear  window r2 t0.05  0.519
  2   728  square       yes   bilinear  window r2 t0.05  0.514
  3   798  square       yes   bilinear  window r1 t0.05  0.513
  4   798  square       yes   bilinear  window r3 t0.05  0.512
  5   896  keep_aspect  yes   nearest   window r1 t0.05  0.510

Writes one CSV per configuration in the competition format (id,prediction).
Features are extracted once per distinct (resolution, resize, flip) setting.

Usage
  python experiments/dino_v2_top5_submissions.py --data_dir data --output_dir submissions
  python experiments/dino_v2_top5_submissions.py --split train   # sanity-check scores on train
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dino_v2_param_testing import (  # noqa: E402
    DEVICE,
    MAIN_THRESHOLD,
    build_feature_cache,
    load_model,
    match,
    sample_source,
)


TOP_CONFIGS = [
    # (rank, resolution, resize_mode, flip_tta, source_method, matching_method, train_pck)
    (1, 798, "square", True, "bilinear", ("window", 2, 0.05), 0.519),
    (2, 728, "square", True, "bilinear", ("window", 2, 0.05), 0.514),
    (3, 798, "square", True, "bilinear", ("window", 1, 0.05), 0.513),
    (4, 798, "square", True, "bilinear", ("window", 3, 0.05), 0.512),
    (5, 896, "keep_aspect", True, "nearest", ("window", 1, 0.05), 0.510),
]


def config_name(rank, resolution, resize_mode, flip_tta, source_method, matching_method):
    matching = "_".join(str(m) for m in matching_method)
    flip = "_flip" if flip_tta else ""
    return f"top{rank}_{resolution}_{resize_mode}{flip}_{source_method}_{matching}"


@torch.inference_mode()
def predict(queries, cache, source_method, matching_method):
    predictions = np.zeros((len(queries), 2))
    for _, group in queries.groupby("pair_id", sort=False):
        source_features, source_size = cache[group["source_image"].iloc[0]]
        target_features, target_size = cache[group["target_image"].iloc[0]]
        source_features = source_features.to(DEVICE).float()
        target_features = target_features.to(DEVICE).float()
        _, grid_h, grid_w = target_features.shape

        points = torch.tensor(
            group[["source_x", "source_y"]].to_numpy(),
            dtype=torch.float32, device=DEVICE,
        )
        vectors = sample_source(source_features, points, source_size, source_method)
        similarity = vectors @ target_features.flatten(1)
        predicted = match(similarity, grid_h, grid_w, target_size, matching_method)
        predictions[group.index.to_numpy()] = predicted.cpu().numpy()
    return predictions


def train_pck(predictions, queries):
    correct = [
        np.linalg.norm(np.asarray(json.loads(t)) - p, axis=1).min() / s < MAIN_THRESHOLD
        for p, t, s in zip(predictions, queries["target_points"], queries["scale"])
    ]
    return float(np.mean(correct))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="submissions")
    parser.add_argument("--model", default="dinov2_vitb14")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    args = parser.parse_args()

    print("Using device:", DEVICE)
    queries = pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
    print(f"{len(queries)} {args.split} queries")

    model = load_model(args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    feature_settings = sorted({(c[1], c[2], c[3]) for c in TOP_CONFIGS})
    for resolution, resize_mode, flip_tta in feature_settings:
        cache = build_feature_cache(
            queries, args.data_dir, model, resolution, resize_mode, flip_tta
        )

        for config in TOP_CONFIGS:
            rank, res, mode, flip, source_method, matching_method, expected = config
            if (res, mode, flip) != (resolution, resize_mode, flip_tta):
                continue

            name = config_name(rank, res, mode, flip, source_method, matching_method)
            predictions = predict(queries, cache, source_method, matching_method)

            if args.split == "train":
                score = train_pck(predictions, queries)
                print(f"{name}: train PCK@0.10 = {score:.4f} (sweep: {expected:.3f})")
                continue

            submission = pd.DataFrame({
                "id": queries["id"],
                "prediction": [
                    json.dumps([round(float(x), 2), round(float(y), 2)])
                    for x, y in predictions
                ],
            })
            path = os.path.join(args.output_dir, f"dinov2_{name}.csv")
            submission.to_csv(path, index=False)
            print(f"Saved {path} ({len(submission)} rows)")

        del cache
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
