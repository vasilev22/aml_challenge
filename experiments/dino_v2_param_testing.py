
import os
import json
import time
import gc

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import wandb

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "/kaggle/working/aml_challenge/data"
MODEL_NAME = "dinov2_vitb14"

RESOLUTIONS = [518, 644, 728, 798, 896]

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
PATCH_SIZE = 14

RESULTS_PATH = "/kaggle/working/aml_challenge/dinov2_benchmark_results.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("Loading DINOv2...")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        MODEL_NAME
    )

    model.eval()
    model.to(DEVICE)

    return model


# ============================================================
# DATA
# ============================================================

def load_validation_data():

    train = pd.read_csv(
        os.path.join(DATA_DIR, "train.csv")
    )

    pair_ids = train["pair_id"].unique()

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(pair_ids)

    split = int(
        len(pair_ids) * (1 - VALIDATION_FRACTION)
    )

    validation_pairs = set(pair_ids[split:])

    validation = train[
        train["pair_id"].isin(validation_pairs)
    ].copy()

    validation["target_points_parsed"] = (
        validation["target_points"].apply(json.loads)
    )

    print(f"Total queries: {len(train)}")
    print(f"Validation queries: {len(validation)}")
    print(f"Validation pairs: {len(validation_pairs)}")

    return validation.reset_index(drop=True)


# ============================================================
# IMAGE FEATURES
# ============================================================

def get_transform(image_size):

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_features(image_path, model, transform):

    with Image.open(image_path) as image:

        image = image.convert("RGB")
        original_size = image.size

        tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        features = model.forward_features(tensor)[
            "x_norm_patchtokens"
        ]

    features = F.normalize(features, dim=-1)

    return features.squeeze(0).cpu().half(), original_size


def build_feature_cache(validation, model, image_size):

    transform = get_transform(image_size)

    image_paths = pd.concat([
        validation["source_image"],
        validation["target_image"]
    ]).unique()

    cache = {}

    for relative_path in tqdm(
        image_paths,
        desc=f"Extracting features at {image_size}px",
        leave=False
    ):

        image_path = os.path.join(
            DATA_DIR,
            relative_path
        )

        features, original_size = extract_features(
            image_path,
            model,
            transform
        )

        cache[relative_path] = (
            features,
            original_size
        )

    return cache


# ============================================================
# SOURCE FEATURE EXTRACTION
# ============================================================

def get_source_vector(
    features,
    x,
    y,
    original_size,
    image_size,
    method
):

    width, height = original_size

    grid_size = image_size // PATCH_SIZE

    resized_x = x * image_size / width
    resized_y = y * image_size / height

    col = int(resized_x // PATCH_SIZE)
    row = int(resized_y // PATCH_SIZE)

    col = np.clip(col, 0, grid_size - 1)
    row = np.clip(row, 0, grid_size - 1)

    if method == "single":

        vector = features[row * grid_size + col].float()

    else:

        radius = 1 if method in ["avg3", "weighted3"] else 2

        vectors = []
        weights = []

        for dy in range(-radius, radius + 1):

            for dx in range(-radius, radius + 1):

                r = row + dy
                c = col + dx

                if 0 <= r < grid_size and 0 <= c < grid_size:

                    vectors.append(
                        features[r * grid_size + c].float()
                    )

                    if method == "weighted3":

                        weights.append(
                            1.0 if dx == 0 and dy == 0 else 0.5
                        )

        vectors = torch.stack(vectors)

        if method == "weighted3":

            weights = torch.tensor(
                weights,
                dtype=torch.float32
            ).unsqueeze(1)

            vector = (
                vectors * weights
            ).sum(dim=0) / weights.sum()

        else:

            vector = vectors.mean(dim=0)

    return F.normalize(vector, dim=0)


# ============================================================
# MATCHING
# ============================================================

def predict_point(
    source_features,
    target_features,
    source_point,
    source_size,
    target_size,
    image_size,
    source_method,
    matching_method
):

    target_width, target_height = target_size

    grid_size = image_size // PATCH_SIZE

    source_vector = get_source_vector(
        source_features,
        source_point[0],
        source_point[1],
        source_size,
        image_size,
        source_method
    )

    target_features = target_features.float()

    similarities = target_features @ source_vector

    k = {
        "argmax": 1,
        "top3": 3,
        "top5": 5,
        "top10": 10
    }[matching_method]

    values, indices = torch.topk(
        similarities,
        k=min(k, len(similarities))
    )

    rows = indices // grid_size
    cols = indices % grid_size

    coords = torch.stack([
        (cols.float() + 0.5) * PATCH_SIZE,
        (rows.float() + 0.5) * PATCH_SIZE
    ], dim=1)

    if matching_method == "argmax":

        predicted = coords[0]

    else:

        weights = torch.softmax(values * 20, dim=0)

        predicted = (
            coords * weights.unsqueeze(1)
        ).sum(dim=0)

    predicted_x = (
        predicted[0].item() * target_width / image_size
    )

    predicted_y = (
        predicted[1].item() * target_height / image_size
    )

    return predicted_x, predicted_y


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    validation,
    cache,
    image_size,
    source_method,
    matching_method
):

    errors = []
    normalized_errors = []

    for _, query in validation.iterrows():

        source_features, source_size = cache[
            query["source_image"]
        ]

        target_features, target_size = cache[
            query["target_image"]
        ]

        predicted_x, predicted_y = predict_point(
            source_features,
            target_features,
            (query["source_x"], query["source_y"]),
            source_size,
            target_size,
            image_size,
            source_method,
            matching_method
        )

        ground_truth = np.asarray(
            query["target_points_parsed"],
            dtype=float
        )

        predicted = np.array([
            predicted_x,
            predicted_y
        ])

        distances = np.linalg.norm(
            ground_truth - predicted,
            axis=1
        )

        error = distances.min()

        diagonal = np.sqrt(
            target_size[0] ** 2 + target_size[1] ** 2
        )

        errors.append(error)
        normalized_errors.append(error / diagonal)

    errors = np.asarray(errors)
    normalized_errors = np.asarray(normalized_errors)

    return {
        "pck_1pct": np.mean(normalized_errors <= 0.01),
        "pck_2_5pct": np.mean(normalized_errors <= 0.025),
        "pck_5pct": np.mean(normalized_errors <= 0.05),
        "mean_error_px": np.mean(errors),
        "median_error_px": np.median(errors)
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    validation,
    cache,
    image_size,
    source_method,
    matching_method
):

    start = time.time()

    metrics = evaluate(
        validation,
        cache,
        image_size,
        source_method,
        matching_method
    )

    elapsed = time.time() - start

    result = {
        "resolution": image_size,
        "source_method": source_method,
        "matching_method": matching_method,
        **metrics,
        "time_seconds": elapsed
    }

    wandb.log({
        "resolution": image_size,
        "source_method": source_method,
        "matching_method": matching_method,
        **metrics,
        "experiment_time": elapsed
    })

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("Using device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    validation = load_validation_data()

    wandb.init(
        project="aml-challenge",
        name="dinov2-hyperparameter-benchmark",
        config={
            "model": MODEL_NAME,
            "resolutions": RESOLUTIONS,
            "validation_fraction": VALIDATION_FRACTION,
            "patch_size": PATCH_SIZE,
            "random_seed": RANDOM_SEED
        }
    )

    model = load_model()

    results = []

    total_experiments = len(RESOLUTIONS) + 4 + 4

    progress = tqdm(
        total=total_experiments,
        desc="Overall benchmark"
    )

    # STAGE 1: RESOLUTION SWEEP

    print("\nSTAGE 1: RESOLUTION SWEEP")

    resolution_results = []

    for image_size in RESOLUTIONS:

        cache = build_feature_cache(
            validation,
            model,
            image_size
        )

        result = run_experiment(
            validation,
            cache,
            image_size,
            "single",
            "argmax"
        )

        results.append(result)
        resolution_results.append(result)

        progress.update(1)

        tqdm.write(
            f"Resolution {image_size}: "
            f"PCK@2.5% = {result['pck_2_5pct']:.4f}"
        )

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    best_resolution = max(
        resolution_results,
        key=lambda x: x["pck_2_5pct"]
    )["resolution"]

    print("\nBest resolution:", best_resolution)

    wandb.log({
        "best_resolution": best_resolution
    })

    # STAGE 2: SOURCE FEATURE SWEEP

    print("\nSTAGE 2: SOURCE FEATURE SWEEP")

    cache = build_feature_cache(
        validation,
        model,
        best_resolution
    )

    source_methods = [
        "single",
        "avg3",
        "weighted3",
        "avg5"
    ]

    source_results = []

    for method in source_methods:

        result = run_experiment(
            validation,
            cache,
            best_resolution,
            method,
            "argmax"
        )

        results.append(result)
        source_results.append(result)

        progress.update(1)

        tqdm.write(
            f"Source method {method}: "
            f"PCK@2.5% = {result['pck_2_5pct']:.4f}"
        )

    best_source = max(
        source_results,
        key=lambda x: x["pck_2_5pct"]
    )["source_method"]

    print("\nBest source method:", best_source)

    wandb.log({
        "best_source_method": best_source
    })

    # STAGE 3: MATCHING SWEEP

    print("\nSTAGE 3: MATCHING SWEEP")

    matching_methods = [
        "argmax",
        "top3",
        "top5",
        "top10"
    ]

    matching_results = []

    for method in matching_methods:

        result = run_experiment(
            validation,
            cache,
            best_resolution,
            best_source,
            method
        )

        results.append(result)
        matching_results.append(result)

        progress.update(1)

        tqdm.write(
            f"Matching {method}: "
            f"PCK@2.5% = {result['pck_2_5pct']:.4f}"
        )

    progress.close()

    # FINAL RESULTS

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "pck_2_5pct",
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    best_result = results_df.iloc[0]

    print("\nBest configuration:")
    print(best_result.to_string())

    wandb.log({
        "final_best_pck_2_5pct": best_result["pck_2_5pct"],
        "final_best_resolution": best_result["resolution"],
        "final_best_source_method": best_result["source_method"],
        "final_best_matching_method": best_result["matching_method"],
        "experiment_results": wandb.Table(dataframe=results_df)
    })

    print("\nResults saved to:", RESULTS_PATH)

    del cache
    del model

    gc.collect()
    torch.cuda.empty_cache()

    wandb.finish()


if __name__ == "__main__":
    main()
