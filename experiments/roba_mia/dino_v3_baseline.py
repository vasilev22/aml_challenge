
import os
import json
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from transformers import AutoModel


IMAGE_SIZE = 512
PATCH_SIZE = 16
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m" #use this later one
#MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

def load_model(device):
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        token=os.environ.get("HF_TOKEN")
    )

    model.eval()
    model.to(device)

    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_features(image_path, model, transform, device):

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_size = image.size

        tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)

        features = outputs.last_hidden_state

        num_register_tokens = getattr(
            model.config,
            "num_register_tokens",
            4
        )

        features = features[
            :,
            1 + num_register_tokens:,
            :
        ]

    features = F.normalize(features, dim=-1)

    return features.squeeze(0).cpu().half(), original_size


def predict_point(
    source_features,
    target_features,
    source_point,
    source_size,
    target_size,
    device
):

    source_width, source_height = source_size
    target_width, target_height = target_size

    x, y = source_point

    resized_x = x * IMAGE_SIZE / source_width
    resized_y = y * IMAGE_SIZE / source_height

    col = int(resized_x // PATCH_SIZE)
    row = int(resized_y // PATCH_SIZE)

    col = np.clip(col, 0, GRID_SIZE - 1)
    row = np.clip(row, 0, GRID_SIZE - 1)

    source_index = row * GRID_SIZE + col

    source_vector = source_features[source_index].to(
        device=device,
        dtype=torch.float32
    )

    target_features = target_features.to(
        device=device,
        dtype=torch.float32
    )

    similarities = target_features @ source_vector

    best_index = similarities.argmax().item()

    best_row = best_index // GRID_SIZE
    best_col = best_index % GRID_SIZE

    predicted_x = (best_col + 0.5) * PATCH_SIZE
    predicted_y = (best_row + 0.5) * PATCH_SIZE

    predicted_x *= target_width / IMAGE_SIZE
    predicted_y *= target_height / IMAGE_SIZE

    return predicted_x, predicted_y


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")

    parser.add_argument(
        "--output",
        default="submissions/dinov3_baseline.csv"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = load_model(device)
    transform = get_transform()

    test = pd.read_csv(
        os.path.join(args.data_dir, "test.csv")
    )

    feature_cache = {}

    def get_features(relative_path):

        if relative_path not in feature_cache:

            image_path = os.path.join(
                args.data_dir,
                relative_path
            )

            features, size = extract_features(
                image_path,
                model,
                transform,
                device
            )

            feature_cache[relative_path] = (
                features,
                size
            )

        return feature_cache[relative_path]

    predictions = []

    for i, query in test.iterrows():

        source_features, source_size = get_features(
            query["source_image"]
        )

        target_features, target_size = get_features(
            query["target_image"]
        )

        x, y = predict_point(
            source_features,
            target_features,
            (query["source_x"], query["source_y"]),
            source_size,
            target_size,
            device
        )

        predictions.append({
            "id": query["id"],
            "prediction": json.dumps([float(x), float(y)])
        })

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(test)}")

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    submission = pd.DataFrame(predictions)

    submission.to_csv(args.output, index=False)

    print("Submission saved:", args.output)
    print("Total predictions:", len(submission))


if __name__ == "__main__":
    main()
