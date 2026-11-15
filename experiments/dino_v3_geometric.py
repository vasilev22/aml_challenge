"""
Geometry-aware matching (affine RANSAC + penalty + zoom) con backend DINOv2 / DINOv3.

Versione autonoma: gli helper che prima venivano da dino_v2_param_testing
(load_model, backbone, sample_source, match, ...) sono definiti qui sotto.
La parte geometrica (estimate_affine, predict, ...) e' invariata rispetto alla versione DINOv2.

Uso
  # sanity check: dovrebbe avvicinarsi ai risultati della tua versione DINOv2
  python dino_v3_geometric.py --data_dir . --model dinov2_vitb14

  # DINOv3
  python dino_v3_geometric.py --data_dir . --model dinov3_vitb16

  # + submission per la config migliore
  python dino_v3_geometric.py --data_dir . --model dinov3_vitb16 --submit

Note
  * La risoluzione viene arrotondata al multiplo della patch del modello
    (798 -> 798 per DINOv2/patch14, 798 -> 800 per DINOv3/patch16).
  * DINOv3 su HuggingFace e' gated: serve il login (huggingface-cli login).
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
MAIN_THRESHOLD = 0.10

FLIP_TTA = True
WINDOW = ("window", 2, 0.05)          # soft-argmax: raggio (in celle), temperatura

LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
BOX_PENALTY = 1.0           # effectively a hard mask (cosine sims live in [-1, 1])

SOURCE_MARGIN = 0.25        # margin around source keypoint bbox, fraction of its longest side
TARGET_MARGIN = 0.15
MIN_BOX_FRAC = 0.25         # boxes are at least this fraction of the image's longest side

RANSAC_ITERS = 2000
INLIER_THRESH = 0.06        # in units of the target image's longest side
MIN_INLIERS = 8


# ============================================================
# BACKBONE (DINOv2 / DINOv3 via HuggingFace)
# ============================================================

HF_NAMES = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


class Backbone:
    def __init__(self, name, resolution=798, layer=-1):
        hf_name = HF_NAMES.get(name, name)             # puoi passare anche un id HF diretto
        self.name = name
        self.model = AutoModel.from_pretrained(hf_name).to(DEVICE).eval()
        self.patch = self.model.config.patch_size
        self.resolution = max(1, round(resolution / self.patch)) * self.patch
        self.layer = layer                              # -1 = ultimo layer (con norm finale)

    @property
    def grid(self):
        return self.resolution // self.patch

    @torch.inference_mode()
    def features(self, x, gh, gw):
        """x: [1,3,H,W] normalizzato -> feature [C, gh, gw] (non normalizzate)."""
        if self.layer == -1:
            out = self.model(pixel_values=x).last_hidden_state
        else:
            out = self.model(pixel_values=x, output_hidden_states=True).hidden_states[self.layer]
        tokens = out[:, -gh * gw:]                      # scarta CLS e register tokens
        return tokens.reshape(gh, gw, -1).permute(2, 0, 1)


def sample_source(features, points, size, mode="bilinear"):
    """Feature (normalizzate) nei punti sorgente. features [C,gh,gw]; points [Q,2] in px della regione."""
    w, h = size
    grid = torch.stack([points[:, 0] / w * 2 - 1, points[:, 1] / h * 2 - 1], dim=1)[None, :, None, :]
    out = F.grid_sample(features[None], grid.to(features.dtype), mode="bilinear",
                        padding_mode="border", align_corners=False)     # [1,C,Q,1]
    return F.normalize(out[0, :, :, 0].T, dim=1)                         # [Q,C]


def match(scores, gh, gw, size, window):
    """Argmax + soft-argmax in una finestra attorno al massimo. scores [Q, gh*gw] -> [Q,2] px (regione)."""
    _, radius, temp = window
    Q = scores.shape[0]
    s = scores.view(Q, gh, gw)
    idx = s.flatten(1).argmax(dim=1)
    iy, ix = idx // gw, idx % gw

    d = torch.arange(-radius, radius + 1, device=s.device)
    ys, xs = iy[:, None] + d[None], ix[:, None] + d[None]                # [Q,K]
    valid = ((ys >= 0) & (ys < gh))[:, :, None] & ((xs >= 0) & (xs < gw))[:, None, :]
    ys, xs = ys.clamp(0, gh - 1), xs.clamp(0, gw - 1)

    q = torch.arange(Q, device=s.device)[:, None, None]
    win = s[q, ys[:, :, None], xs[:, None, :]]                           # [Q,K,K]
    win = (win / temp).masked_fill(~valid, float("-inf"))
    wgt = torch.softmax(win.flatten(1), dim=1).view_as(win)

    cy = (wgt * ys[:, :, None].float()).sum(dim=(1, 2))
    cx = (wgt * xs[:, None, :].float()).sum(dim=(1, 2))
    w, h = size
    return torch.stack([(cx + 0.5) / gw * w, (cy + 0.5) / gh * h], dim=1)


# ============================================================
# FEATURES
# ============================================================

def load_image(path):
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB"))          # [3, H, W]


@torch.inference_mode()
def extract(model, image, box=None):
    """Features of image (or of crop box=(x0,y0,x1,y1)). Returns ([C,g,g] on DEVICE, region)."""
    _, height, width = image.shape
    if box is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = image[:, y0:y1, x0:x1]
    x = TF.resize(crop, [model.resolution, model.resolution], antialias=True)
    x = TF.normalize(x, NORM_MEAN, NORM_STD).unsqueeze(0).to(DEVICE)
    g = model.grid
    features = model.features(x, g, g)
    if FLIP_TTA:
        flipped = model.features(x.flip(-1), g, g).flip(-1)
        features = F.normalize(features, dim=0) + F.normalize(flipped, dim=0)
    return F.normalize(features, dim=0), (x0, y0, x1 - x0, y1 - y0)


def global_cache(model, queries, data_dir, cache_path):
    if cache_path and os.path.exists(cache_path):
        print("Loading cached features:", cache_path)
        return torch.load(cache_path)
    cache = {}
    paths = pd.concat([queries["source_image"], queries["target_image"]]).unique()
    for path in tqdm(paths, desc="global features"):
        features, _ = extract(model, load_image(os.path.join(data_dir, path)))
        cache[path] = features.half().cpu()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    return cache


# ============================================================
# GEOMETRY  (invariata)
# ============================================================

def grid_centers(grid_h, grid_w, region):
    """Pixel centres (original image coords) of a feature grid covering region=(x0,y0,w,h)."""
    x0, y0, w, h = region
    xs = x0 + (torch.arange(grid_w, device=DEVICE) + 0.5) / grid_w * w
    ys = y0 + (torch.arange(grid_h, device=DEVICE) + 0.5) / grid_h * h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=1)


def expand_box(lo, hi, size, margin):
    """Box around [lo, hi] with min size and margin, clipped to the image, or None if degenerate."""
    width, height = size
    min_side = MIN_BOX_FRAC * max(width, height)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    w = max(hi[0] - lo[0], min_side)
    h = max(hi[1] - lo[1], min_side)
    pad = margin * max(w, h)
    x0, x1 = max(0.0, cx - w / 2 - pad), min(float(width), cx + w / 2 + pad)
    y0, y1 = max(0.0, cy - h / 2 - pad), min(float(height), cy + h / 2 + pad)
    if x1 - x0 < 0.5 * min_side or y1 - y0 < 0.5 * min_side:
        return None
    return (x0, y0, x1, y1)


def ones(n):
    return torch.ones(n, 1, device=DEVICE)


def estimate_affine(fs, ft, source_size, target_size, source_box, generator):
    """MNN matches from inside source_box + RANSAC affine. Returns callable pts->pts or None."""
    _, gh, gw = fs.shape
    similarity = fs.flatten(1).T @ ft.flatten(1)                      # [Ns, Nt]
    forward = similarity.argmax(dim=1)
    backward = similarity.argmax(dim=0)
    mutual = backward[forward] == torch.arange(len(forward), device=DEVICE)

    ps = grid_centers(gh, gw, (0, 0, *source_size))
    pt = grid_centers(*ft.shape[1:], (0, 0, *target_size))
    x0, y0, x1, y1 = source_box
    inside = (ps[:, 0] >= x0) & (ps[:, 0] <= x1) & (ps[:, 1] >= y0) & (ps[:, 1] <= y1)
    keep = mutual & inside
    if keep.sum() < MIN_INLIERS:
        return None

    ns, nt = float(max(source_size)), float(max(target_size))
    P = ps[keep] / ns
    Q = pt[forward[keep]] / nt
    m = len(P)
    X = torch.cat([P, ones(m)], dim=1)                                # [M, 3]

    idx = torch.randint(0, m, (RANSAC_ITERS, 3), device=DEVICE, generator=generator)
    B = torch.linalg.pinv(X[idx]) @ Q[idx]                            # [K, 3, 2]
    A = B[:, :2, :].transpose(1, 2)                                   # [K, 2, 2]
    sv = torch.linalg.svdvals(A)
    valid = (sv[:, 1] > 0.1) & (sv[:, 0] < 10)

    err = (X[None] @ B - Q[None]).norm(dim=-1)                        # [K, M]
    inliers = err < INLIER_THRESH
    count = inliers.sum(dim=1).masked_fill(~valid, 0)
    best = count.argmax()
    mask = inliers[best]
    if count[best] < MIN_INLIERS:
        return None

    B = torch.linalg.pinv(X[mask]) @ Q[mask]                          # [3, 2]
    sv = torch.linalg.svdvals(B[:2].T)
    if sv[1] < 0.1 or sv[0] > 10:
        return None

    def transform(points):
        return (torch.cat([points / ns, ones(len(points))], dim=1) @ B) * nt

    return transform


# ============================================================
# PREDICTION  (invariata)
# ============================================================

def config_list():
    configs = []
    for lam in LAMBDAS:
        for box in [False, True]:
            configs.append(("global", lam, box))
        configs.append(("zoom", lam, False))
    return configs


def config_name(config):
    mode, lam, box = config
    return f"{mode}_lam{lam}" + ("_box" if box else "")


@torch.inference_mode()
def predict(queries, pairs, cache, model, data_dir, configs, seed=0, debug=None):
    """debug: optional dict, filled with per-pair affine/box info for error analysis."""
    predictions = {c: np.zeros((len(queries), 2)) for c in configs}
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    need_zoom = any(c[0] == "zoom" for c in configs)
    found = 0

    groups = list(queries.groupby("pair_id", sort=False))
    for pair_id, group in tqdm(groups, desc="pairs"):
        meta = pairs[pair_id]
        source_size, target_size = tuple(meta["source_size"]), tuple(meta["target_size"])
        fs = cache[meta["source_image"]].to(DEVICE).float()
        ft = cache[meta["target_image"]].to(DEVICE).float()

        keypoints = np.asarray(meta["source_xy"], dtype=np.float64)
        source_box = expand_box(keypoints.min(0), keypoints.max(0), source_size, SOURCE_MARGIN)
        if source_box is None:
            source_box = (0, 0, *source_size)

        transform = estimate_affine(fs, ft, source_size, target_size, source_box, generator)

        points = torch.tensor(
            group[["source_x", "source_y"]].to_numpy(), dtype=torch.float32, device=DEVICE
        )

        mu, target_box, object_size = None, None, None
        if transform is not None:
            found += 1
            mu = transform(points)
            sx0, sy0, sx1, sy1 = source_box
            corners = transform(torch.tensor(
                [[sx0, sy0], [sx1, sy0], [sx0, sy1], [sx1, sy1]],
                dtype=torch.float32, device=DEVICE,
            )).cpu().numpy()
            lo, hi = corners.min(0), corners.max(0)
            target_box = expand_box(lo, hi, target_size, TARGET_MARGIN)
            object_size = float(np.clip(
                (hi - lo).max(), 0.1 * max(target_size), 1.5 * max(target_size)
            ))

        if debug is not None:
            debug[pair_id] = {
                "affine": transform is not None,
                "source_box": source_box,
                "target_box": target_box,
            }

        views = {"global": (fs, (0, 0, *source_size), ft, (0, 0, *target_size))}
        if need_zoom:
            if target_box is not None:
                source_image = load_image(os.path.join(data_dir, meta["source_image"]))
                target_image = load_image(os.path.join(data_dir, meta["target_image"]))
                zs, rs = extract(model, source_image, source_box)
                zt, rt = extract(model, target_image, target_box)
                views["zoom"] = (zs, rs, zt, rt)
            else:
                views["zoom"] = views["global"]

        rows = group.index.to_numpy()
        for mode, (fsv, rs, ftv, rt) in views.items():
            _, gh, gw = ftv.shape
            local = points - torch.tensor([rs[0], rs[1]], device=DEVICE, dtype=torch.float32)
            vectors = sample_source(fsv, local, (rs[2], rs[3]), "bilinear")
            similarity = vectors @ ftv.flatten(1)                     # [Q, N]

            if mu is not None:
                centers = grid_centers(gh, gw, rt)
                d2 = ((centers[None] - mu[:, None]) ** 2).sum(-1) / object_size ** 2
                if target_box is not None:
                    bx0, by0, bx1, by1 = target_box
                    outside = ~(
                        (centers[:, 0] >= bx0) & (centers[:, 0] <= bx1)
                        & (centers[:, 1] >= by0) & (centers[:, 1] <= by1)
                    )
                else:
                    outside = torch.zeros(len(centers), dtype=torch.bool, device=DEVICE)

            offset = torch.tensor([rt[0], rt[1]], device=DEVICE, dtype=torch.float32)
            for config in configs:
                if config[0] != mode:
                    continue
                _, lam, box = config
                scores = similarity
                if mu is not None:
                    scores = scores - lam * d2
                    if box:
                        scores = scores - BOX_PENALTY * outside.float()[None]
                predicted = match(scores, gh, gw, (rt[2], rt[3]), WINDOW) + offset
                predictions[config][rows] = predicted.cpu().numpy()

    print(f"Affine found for {found}/{len(groups)} pairs")
    return predictions


def pck(predictions, queries):
    errors = np.array([
        np.linalg.norm(np.asarray(json.loads(t)) - p, axis=1).min() / s
        for p, t, s in zip(predictions, queries["target_points"], queries["scale"])
    ])
    return float((errors < MAIN_THRESHOLD).mean()), float((errors < 0.05).mean())


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--submission_dir", default="submissions")
    parser.add_argument("--model", default="dinov3_vitb16")
    parser.add_argument("--resolution", type=int, default=798)
    parser.add_argument("--layer", type=int, default=-1, help="-1 = ultimo layer")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    model = Backbone(args.model, args.resolution, args.layer)
    tag = f"{args.model}_{model.resolution}_L{args.layer}_flip{int(FLIP_TTA)}"
    print(f"Backbone {args.model} | patch {model.patch} | resolution {model.resolution} | grid {model.grid}x{model.grid}")

    def load_split(split):
        queries = pd.read_csv(os.path.join(args.data_dir, f"{split}.csv"))
        pairs = {
            p["pair_id"]: p
            for p in json.load(open(os.path.join(args.data_dir, f"{split}_pairs.json")))["pairs"]
        }
        cache = global_cache(
            model, queries, args.data_dir, os.path.join(args.cache_dir, f"{split}_{tag}.pt")
        )
        return queries, pairs, cache

    queries, pairs, cache = load_split("train")
    configs = config_list()
    predictions = predict(queries, pairs, cache, model, args.data_dir, configs)

    rows = []
    for config in configs:
        p10, p05 = pck(predictions[config], queries)
        rows.append({"config": config_name(config), "mode": config[0], "lambda": config[1],
                     "box": config[2], "pck@0.10": p10, "pck@0.05": p05})
    table = pd.DataFrame(rows).sort_values("pck@0.10", ascending=False)
    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(os.path.join(args.output_dir, f"{tag}_geometric_results.csv"), index=False)
    print(table.to_string(index=False, float_format="%.4f"))

    if args.submit:
        best_name = table.iloc[0]["config"]
        best = next(c for c in configs if config_name(c) == best_name)
        print("Writing test submission for", best_name)
        del cache
        test_queries, test_pairs, test_cache = load_split("test")
        test_pred = predict(test_queries, test_pairs, test_cache, model, args.data_dir, [best])[best]
        os.makedirs(args.submission_dir, exist_ok=True)
        path = os.path.join(args.submission_dir, f"{tag}_geo_{best_name}.csv")
        pd.DataFrame({
            "id": test_queries["id"],
            "prediction": [json.dumps([round(float(x), 2), round(float(y), 2)]) for x, y in test_pred],
        }).to_csv(path, index=False)
        print("Saved", path)


if __name__ == "__main__":
    main()
