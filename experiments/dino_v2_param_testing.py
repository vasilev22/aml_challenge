"""
DINOv2 hyperparameter sweep for the AML semantic-correspondence challenge.

Metric (identical to the leaderboard):
    e(q) = min_{t in targets} ||pred - t|| / scale      correct if e(q) < 0.10
`scale` (train.csv only) is the longest side of the target object's bounding box.
The bounding boxes themselves are NOT provided anywhere; `scale` is all we get.

Sweep structure
  feature configs (expensive, one backbone pass per image):
      resolution x resize mode x flip-TTA
  query configs (cheap, reuse cached features):
      source sampling x matching method

Nothing is trained here, so by default every training query is used for
evaluation (--split all). Use --split val for a held-out subset once you start
fitting something on train.

Examples
  python dino_v2_param_testing.py                         # full sweep, Kaggle paths
  python dino_v2_param_testing.py --data_dir data --model dinov2_vits14 \
      --resolutions 224 --max_pairs 10                    # quick local smoke test
"""

import os
import gc
import json
import time
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = 14
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20

RESOLUTIONS = [518, 644, 728, 798, 896]
RESIZE_MODES = ["square", "keep_aspect"]      # keep_aspect: long side = resolution
FLIP_TTA = [False, True]

SOURCE_METHODS = ["nearest", "bilinear", "avg3"]

# ("argmax",)                    hard argmax, patch centre
# ("window", radius, temp)       soft-argmax restricted to a (2r+1)^2 window around the argmax
# ("topk", k, temp)              soft-argmax over the global top-k (old method; can average
#                                between two accepted targets, kept for comparison)
MATCHING_METHODS = [
    ("argmax",),
    ("window", 1, 0.02),
    ("window", 1, 0.05),
    ("window", 2, 0.02),
    ("window", 2, 0.05),
    ("window", 3, 0.05),
    ("topk", 5, 0.05),
]

PCK_THRESHOLDS = [0.05, 0.10, 0.15]
MAIN_THRESHOLD = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def method_name(method):
    return "_".join(str(m) for m in method)


# ============================================================
# DATA
# ============================================================

def load_queries(data_dir, split, max_pairs=None):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train["targets"] = train["target_points"].apply(
        lambda s: np.asarray(json.loads(s), dtype=np.float64)
    )

    # Object-size buckets on the full train set, so they are comparable across splits.
    train["size_bucket"] = pd.qcut(
        train["scale"], 3, labels=["small", "medium", "large"]
    )

    if split == "val":
        pair_ids = train["pair_id"].unique()
        rng = np.random.default_rng(RANDOM_SEED)
        rng.shuffle(pair_ids)
        keep = set(pair_ids[int(len(pair_ids) * (1 - VALIDATION_FRACTION)):])
        train = train[train["pair_id"].isin(keep)]

    if max_pairs is not None:
        keep = train["pair_id"].drop_duplicates().iloc[:max_pairs]
        train = train[train["pair_id"].isin(set(keep))]

    train = train.reset_index(drop=True)
    print(
        f"Evaluating on {len(train)} queries / "
        f"{train['pair_id'].nunique()} pairs (split={split})"
    )
    return train


# ============================================================
# FEATURES
# ============================================================

def load_model(model_name):
    print(f"Loading {model_name}...")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    return model.eval().to(DEVICE)


def resized_shape(width, height, resolution, mode):
    """Return (new_w, new_h), both multiples of PATCH_SIZE."""
    if mode == "square":
        return resolution, resolution
    s = resolution / max(width, height)
    new_w = max(PATCH_SIZE, round(width * s / PATCH_SIZE) * PATCH_SIZE)
    new_h = max(PATCH_SIZE, round(height * s / PATCH_SIZE) * PATCH_SIZE)
    return new_w, new_h


def backbone(model, x, grid_h, grid_w):
    tokens = model.forward_features(x)["x_norm_patchtokens"]   # [1, H*W, C]
    return tokens[0].T.reshape(-1, grid_h, grid_w).float()      # [C, H, W]


@torch.inference_mode()
def extract_features(image_path, model, resolution, mode, flip_tta):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        tensor = TF.to_tensor(image)

    new_w, new_h = resized_shape(width, height, resolution, mode)
    x = TF.resize(tensor, [new_h, new_w], antialias=True)
    x = TF.normalize(x, NORM_MEAN, NORM_STD).unsqueeze(0).to(DEVICE)

    grid_h, grid_w = new_h // PATCH_SIZE, new_w // PATCH_SIZE
    features = backbone(model, x, grid_h, grid_w)

    if flip_tta:
        flipped = backbone(model, x.flip(-1), grid_h, grid_w).flip(-1)
        features = F.normalize(features, dim=0) + F.normalize(flipped, dim=0)

    features = F.normalize(features, dim=0)
    return features.half().cpu(), (width, height)


def build_feature_cache(queries, data_dir, model, resolution, mode, flip_tta):
    paths = pd.concat([queries["source_image"], queries["target_image"]]).unique()
    cache = {}
    desc = f"features res={resolution} {mode} flip={flip_tta}"
    for relative_path in tqdm(paths, desc=desc, leave=False):
        cache[relative_path] = extract_features(
            os.path.join(data_dir, relative_path),
            model, resolution, mode, flip_tta,
        )
    return cache


# ============================================================
# SOURCE SAMPLING
# ============================================================

def sample_source(features, points, size, method):
    """
    features: [C, H, W] on DEVICE, points: [Q, 2] original pixels -> [Q, C] unit vectors.

    grid_sample with align_corners=False maps x/width in [0, 1] onto the feature grid
    with patch centres at (i + 0.5) / W, which matches how the patches tile the image
    for both resize modes.
    """
    width, height = size
    if method == "avg3":
        features = F.avg_pool2d(
            features[None], 3, stride=1, padding=1, count_include_pad=False
        )[0]
    grid = torch.stack(
        [points[:, 0] / width * 2 - 1, points[:, 1] / height * 2 - 1], dim=-1
    ).view(1, 1, -1, 2)
    vectors = F.grid_sample(
        features[None], grid,
        mode="nearest" if method == "nearest" else "bilinear",
        padding_mode="border", align_corners=False,
    )[0, :, 0].T
    return F.normalize(vectors, dim=-1)


# ============================================================
# MATCHING
# ============================================================

def match(similarity, grid_h, grid_w, size, method):
    """similarity: [Q, H*W] -> predicted points [Q, 2] in original target pixels."""
    width, height = size
    kind = method[0]

    if kind == "argmax":
        index = similarity.argmax(dim=1)
        rows = (index // grid_w).float()
        cols = (index % grid_w).float()

    elif kind == "window":
        _, radius, temperature = method
        q = similarity.shape[0]
        sim = similarity.view(q, grid_h, grid_w)
        index = similarity.argmax(dim=1)
        r0, c0 = index // grid_w, index % grid_w

        offsets = torch.arange(-radius, radius + 1, device=similarity.device)
        k = len(offsets)
        rows = (r0[:, None, None] + offsets[None, :, None]).expand(q, k, k)
        cols = (c0[:, None, None] + offsets[None, None, :]).expand(q, k, k)
        valid = (rows >= 0) & (rows < grid_h) & (cols >= 0) & (cols < grid_w)

        q_index = torch.arange(q, device=similarity.device)[:, None, None]
        values = sim[q_index, rows.clamp(0, grid_h - 1), cols.clamp(0, grid_w - 1)]
        values = values.masked_fill(~valid, float("-inf"))

        weights = torch.softmax((values / temperature).flatten(1), dim=1).view(q, k, k)
        rows = (weights * rows.float()).sum(dim=(1, 2))
        cols = (weights * cols.float()).sum(dim=(1, 2))

    elif kind == "topk":
        _, k, temperature = method
        values, index = similarity.topk(min(k, similarity.shape[1]), dim=1)
        weights = torch.softmax(values / temperature, dim=1)
        rows = (weights * (index // grid_w).float()).sum(dim=1)
        cols = (weights * (index % grid_w).float()).sum(dim=1)

    else:
        raise ValueError(f"Unknown matching method {method}")

    x = (cols + 0.5) / grid_w * width
    y = (rows + 0.5) / grid_h * height
    return torch.stack([x, y], dim=1)


# ============================================================
# EVALUATION
# ============================================================

def predict_all_configs(queries, cache):
    """Run every (source method, matching method) combination in one pass over pairs."""
    configs = list(itertools.product(SOURCE_METHODS, MATCHING_METHODS))
    predictions = {c: np.zeros((len(queries), 2)) for c in configs}

    for _, group in queries.groupby("pair_id", sort=False):
        row_index = group.index.to_numpy()
        source_features, source_size = cache[group["source_image"].iloc[0]]
        target_features, target_size = cache[group["target_image"].iloc[0]]

        source_features = source_features.to(DEVICE).float()
        target_features = target_features.to(DEVICE).float()
        _, grid_h, grid_w = target_features.shape
        target_flat = target_features.flatten(1)                     # [C, H*W]

        points = torch.tensor(
            group[["source_x", "source_y"]].to_numpy(),
            dtype=torch.float32, device=DEVICE,
        )

        for source_method in SOURCE_METHODS:
            vectors = sample_source(source_features, points, source_size, source_method)
            similarity = vectors @ target_flat                       # [Q, H*W]
            for matching_method in MATCHING_METHODS:
                predicted = match(similarity, grid_h, grid_w, target_size, matching_method)
                predictions[(source_method, matching_method)][row_index] = (
                    predicted.cpu().numpy()
                )

    return predictions


def normalized_errors(predictions, queries):
    return np.array([
        np.linalg.norm(targets - prediction, axis=1).min() / scale
        for prediction, targets, scale in zip(
            predictions, queries["targets"], queries["scale"]
        )
    ])


def summarize(errors, queries):
    correct = errors < MAIN_THRESHOLD
    n = len(errors)
    p = correct.mean()
    metrics = {f"pck@{t:.2f}": float((errors < t).mean()) for t in PCK_THRESHOLDS}
    metrics["pck_stderr"] = float(np.sqrt(p * (1 - p) / max(n, 1)))
    metrics["median_error"] = float(np.median(errors))
    for bucket in ["small", "medium", "large"]:
        mask = (queries["size_bucket"] == bucket).to_numpy()
        metrics[f"pck@0.10_{bucket}"] = float(correct[mask].mean()) if mask.any() else np.nan
    multi = (queries["targets"].apply(len) > 1).to_numpy()
    metrics["pck@0.10_single_target"] = float(correct[~multi].mean()) if (~multi).any() else np.nan
    metrics["pck@0.10_multi_target"] = float(correct[multi].mean()) if multi.any() else np.nan
    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/kaggle/working/aml_challenge/data")
    parser.add_argument("--output_dir", default="/kaggle/working/aml_challenge")
    parser.add_argument("--model", default="dinov2_vitb14")
    parser.add_argument("--split", choices=["all", "val"], default="all")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS)
    parser.add_argument("--resize_modes", nargs="+", default=RESIZE_MODES)
    parser.add_argument("--flip_tta", type=int, nargs="+", default=[int(f) for f in FLIP_TTA])
    parser.add_argument("--max_pairs", type=int, default=None, help="subset for smoke tests")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    print("Using device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{args.model}_sweep_results.csv")
    best_queries_path = os.path.join(args.output_dir, f"{args.model}_best_per_query.csv")

    queries = load_queries(args.data_dir, args.split, args.max_pairs)
    model = load_model(args.model)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project="aml-challenge",
            name=f"{args.model}-sweep",
            config={**vars(args), "patch_size": PATCH_SIZE},
        )

    results = []
    best = {"pck@0.10": -1.0}

    feature_configs = list(itertools.product(
        args.resolutions, args.resize_modes, [bool(f) for f in args.flip_tta]
    ))

    for resolution, resize_mode, flip_tta in tqdm(feature_configs, desc="feature configs"):
        start = time.time()
        cache = build_feature_cache(
            queries, args.data_dir, model, resolution, resize_mode, flip_tta
        )
        extraction_time = time.time() - start

        all_predictions = predict_all_configs(queries, cache)

        for (source_method, matching_method), predictions in all_predictions.items():
            errors = normalized_errors(predictions, queries)
            result = {
                "model": args.model,
                "resolution": resolution,
                "resize_mode": resize_mode,
                "flip_tta": flip_tta,
                "source_method": source_method,
                "matching_method": method_name(matching_method),
                **summarize(errors, queries),
                "extraction_time_s": extraction_time,
            }
            results.append(result)
            if run is not None:
                run.log(result)

            if result["pck@0.10"] > best["pck@0.10"]:
                best = result
                per_query = queries[["id", "pair_id", "scale", "size_bucket"]].copy()
                per_query["pred_x"] = predictions[:, 0]
                per_query["pred_y"] = predictions[:, 1]
                per_query["error"] = errors
                per_query["correct"] = errors < MAIN_THRESHOLD
                per_query.to_csv(best_queries_path, index=False)

        # Save after every feature config so a Kaggle timeout keeps partial results.
        pd.DataFrame(results).to_csv(results_path, index=False)

        top = max(
            (r for r in results if r["resolution"] == resolution
             and r["resize_mode"] == resize_mode and r["flip_tta"] == flip_tta),
            key=lambda r: r["pck@0.10"],
        )
        tqdm.write(
            f"res={resolution:4d} {resize_mode:11s} flip={int(flip_tta)} | "
            f"best PCK@0.10={top['pck@0.10']:.4f} (+/-{top['pck_stderr']:.4f}) "
            f"with {top['source_method']} + {top['matching_method']}"
        )

        del cache, all_predictions
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(results).sort_values("pck@0.10", ascending=False)
    columns = [
        "resolution", "resize_mode", "flip_tta", "source_method", "matching_method",
        "pck@0.05", "pck@0.10", "pck@0.15",
        "pck@0.10_small", "pck@0.10_medium", "pck@0.10_large",
        "pck@0.10_multi_target",
    ]
    print("\nTop 15 configurations:")
    print(table[columns].head(15).to_string(index=False, float_format="%.4f"))
    print(f"\nResults: {results_path}")
    print(f"Per-query errors of best config: {best_queries_path}")

    if run is not None:
        run.summary.update({f"best_{k}": v for k, v in best.items()})
        run.finish()


if __name__ == "__main__":
    main()
