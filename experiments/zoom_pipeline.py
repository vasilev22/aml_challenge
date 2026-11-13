"""
Zoom pipeline, stage 1: per-query similarity maps (the expensive, GPU part).

For each pair:
  * global features (full image, square resize, flip TTA) of source and target
  * a RANSAC affine fit from mutual nearest-neighbour matches, always computed with the
    reference backbone (DINOv2-B) so every backbone uses identical crops
  * object crops: source = bbox of the pair's source keypoints (+ margin); target = that box
    mapped through the affine (+ margin). Falls back to the full images if no affine is found.
  * for every query, two similarity maps on the target-crop grid:
        zoom : source vector from the source crop    vs  target-crop features
        glob : source vector from the full source    vs  full target features, resampled onto
               the crop grid
Maps are saved in fp16, so blending, matching settings and backbone ensembles can be evaluated
in seconds by evaluate_maps.py without the GPU.

Usage
  python experiments/zoom_pipeline.py --split train
  python experiments/zoom_pipeline.py --split test
  python experiments/zoom_pipeline.py --split train --tgt_margin 0.05 --square
"""

import os
import sys
import json
import time
import types
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dino_v2_param_testing import sample_source  # noqa: E402
from dino_v2_geometric import estimate_affine, grid_centers  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REFERENCE_BACKBONE = "dinov2_vitb14"
REFERENCE_RES = 798
AFFINE_BOX = dict(margin=0.25, min_frac=0.25, square=False)   # fixed: selects matches for the affine fit

DINOV3_REPOS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

HF_DINOV2_REPOS = {
    "dinov2_vitl14_hf": "facebook/dinov2-large",
    "dinov2_vitg14_hf": "facebook/dinov2-giant",
}

# Stable Diffusion (DIFT-style features): one noisy U-Net pass, features of decoder block
# SD_UP_INDEX (stride 16), averaged over SD_ENSEMBLE noise draws. Community mirrors of the
# original weights; safetensors only.
SD_REPOS = {
    "sd21": "sd2-community/stable-diffusion-2-1",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}
SD_TIMESTEP = int(os.environ.get("SD_TIMESTEP", 261))
SD_UP_INDEX = int(os.environ.get("SD_UP_INDEX", 1))
SD_ENSEMBLE = int(os.environ.get("SD_ENSEMBLE", 2))


class _StopForward(Exception):
    """Raised by a hook to skip the rest of the U-Net once the wanted features are captured."""


# ============================================================
# BACKBONES
# ============================================================

def _sdpa_attention(self, x, attn_bias=None):
    """Drop-in for DINOv2's Attention.forward using PyTorch's fused attention kernel."""
    b, n, c = x.shape
    qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
    return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, c)))


class Backbone:
    """Callable: normalised images [B,3,H,W] -> patch features [B,C,H/p,W/p] (float32)."""

    def __init__(self, name, fp16=True):
        self.name = name
        self.fp16 = fp16 and DEVICE.type == "cuda"
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        if name in HF_DINOV2_REPOS:
            # same weights via Hugging Face (used when Meta's download server is unreachable)
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(HF_DINOV2_REPOS[name], attn_implementation="sdpa")
            self.patch = 14
            self.skip = 1
            self.kind = "dinov3"          # same calling convention: last_hidden_state minus prefix tokens
        elif name.startswith("dinov2"):
            self.model = torch.hub.load("facebookresearch/dinov2", name)
            for module in self.model.modules():
                if hasattr(module, "qkv") and hasattr(module, "num_heads"):
                    module.forward = types.MethodType(_sdpa_attention, module)
            self.patch = 14
            self.kind = "dinov2"
        elif name.startswith("dinov3"):
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                DINOV3_REPOS[name], token=os.environ.get("HF_TOKEN")
            )
            self.patch = 16
            self.skip = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
            self.kind = "dinov3"
        elif name in SD_REPOS:
            from diffusers import StableDiffusionPipeline, DDPMScheduler
            pipe = StableDiffusionPipeline.from_pretrained(
                SD_REPOS[name], torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
                safety_checker=None, requires_safety_checker=False,
            )
            tokens = pipe.tokenizer([""], padding="max_length", return_tensors="pt",
                                    max_length=pipe.tokenizer.model_max_length)
            text_encoder = pipe.text_encoder.to(DEVICE)
            with torch.inference_mode():
                self.text = text_encoder(tokens.input_ids.to(DEVICE))[0]
            del text_encoder
            self.vae = pipe.vae.to(DEVICE).eval()
            self.unet = pipe.unet.to(DEVICE).eval()
            self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            del pipe
            self.unet.up_blocks[SD_UP_INDEX].register_forward_hook(self._capture)
            self.model = self.unet
            self.patch = 16
            self.mean, self.std = [0.5] * 3, [0.5] * 3          # images in [-1, 1]
            self.kind = "sd"
        else:
            raise ValueError(f"Unknown backbone {name}")
        self.model.eval().to(DEVICE)

    def _capture(self, module, inputs, output):
        self._features = output
        raise _StopForward

    def _sd_features(self, x):
        latents = self.vae.encode(x.half()).latent_dist.mean * self.vae.config.scaling_factor
        t = torch.tensor([SD_TIMESTEP], device=DEVICE)
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        total = 0
        for _ in range(SD_ENSEMBLE):
            noise = torch.randn(latents.shape, generator=generator, device=DEVICE, dtype=latents.dtype)
            noisy = self.scheduler.add_noise(latents, noise, t)
            try:
                self.unet(noisy, t, encoder_hidden_states=self.text.expand(len(x), -1, -1))
            except _StopForward:
                pass
            total = total + self._features.float()
        return total / SD_ENSEMBLE

    @torch.inference_mode()
    def __call__(self, x):
        b, _, height, width = x.shape
        h, w = height // self.patch, width // self.patch
        if self.kind == "sd":
            features = self._sd_features(x)
            return F.interpolate(features, size=(h, w), mode="bilinear", align_corners=False) \
                if features.shape[-2:] != (h, w) else features
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            if self.kind == "dinov2":
                tokens = self.model.forward_features(x)["x_norm_patchtokens"]
            else:
                tokens = self.model(pixel_values=x).last_hidden_state[:, self.skip:]
        return tokens.float().transpose(1, 2).reshape(b, -1, h, w)


# ============================================================
# IMAGES, CROPS, FEATURES
# ============================================================

def load_image(path):
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB"))


def preprocess(image, box, res, mean, std):
    """Crop (box in pixels or None) and resize to res x res. Returns (tensor, region x0,y0,w,h)."""
    _, height, width = image.shape
    if box is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
    x = TF.resize(image[:, y0:y1, x0:x1], [res, res], antialias=True)
    return TF.normalize(x, mean, std), (x0, y0, x1 - x0, y1 - y0)


@torch.inference_mode()
def extract_many(backbone, items, res):
    """items: list of (image, box). Flip TTA. Returns list of ([C,h,w] on DEVICE, region)."""
    tensors, regions = [], []
    for image, box in items:
        x, region = preprocess(image, box, res, backbone.mean, backbone.std)
        tensors += [x, x.flip(-1)]
        regions.append(region)
    features = backbone(torch.stack(tensors).to(DEVICE))
    out = []
    for i, region in enumerate(regions):
        f = F.normalize(features[2 * i], dim=0) + F.normalize(features[2 * i + 1].flip(-1), dim=0)
        out.append((F.normalize(f, dim=0), region))
    return out


def make_box(lo, hi, size, margin, min_frac, square):
    """Box around [lo, hi]: min size, margin, clipped to the image; optionally made square
    (shifted to stay inside the image). Returns None if the box is degenerate."""
    width, height = size
    min_side = min_frac * max(width, height)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    w = max(hi[0] - lo[0], min_side)
    h = max(hi[1] - lo[1], min_side)
    pad = margin * max(w, h)
    x0, x1 = max(0.0, cx - w / 2 - pad), min(float(width), cx + w / 2 + pad)
    y0, y1 = max(0.0, cy - h / 2 - pad), min(float(height), cy + h / 2 + pad)
    if x1 - x0 < 0.5 * min_side or y1 - y0 < 0.5 * min_side:
        return None
    if square:
        side = max(x1 - x0, y1 - y0)
        sw, sh = min(side, float(width)), min(side, float(height))
        ccx, ccy = (x0 + x1) / 2, (y0 + y1) / 2
        x0 = min(max(ccx - sw / 2, 0.0), width - sw)
        y0 = min(max(ccy - sh / 2, 0.0), height - sh)
        x1, y1 = x0 + sw, y0 + sh
    return (float(x0), float(y0), float(x1), float(y1))


def crop_name(args):
    shape = "sq" if args.square else "rect"
    return f"s{args.src_margin}_t{args.tgt_margin}_m{args.min_frac}_{shape}"


def global_features(backbone, res, paths, data_dir, cache_path):
    if os.path.exists(cache_path):
        print("Loading cached global features:", cache_path)
        return torch.load(cache_path)
    cache = {}
    for path in tqdm(paths, desc=f"global features {backbone.name}"):
        (features, _), = extract_many(backbone, [(load_image(os.path.join(data_dir, path)), None)], res)
        cache[path] = features.half().cpu()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def compute_boxes(queries, pairs, ref_cache, args):
    """Affine (reference backbone) + crop boxes per pair."""
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    boxes = {}
    for pair_id, _ in tqdm(queries.groupby("pair_id", sort=False), desc="affine + boxes"):
        meta = pairs[pair_id]
        ss, ts = tuple(meta["source_size"]), tuple(meta["target_size"])
        keypoints = np.asarray(meta["source_xy"], dtype=np.float64)
        lo, hi = keypoints.min(0), keypoints.max(0)

        affine_box = make_box(lo, hi, ss, **AFFINE_BOX) or (0.0, 0.0, float(ss[0]), float(ss[1]))
        fs = ref_cache[meta["source_image"]].to(DEVICE).float()
        ft = ref_cache[meta["target_image"]].to(DEVICE).float()
        transform = estimate_affine(fs, ft, ss, ts, affine_box, generator)

        source_box = make_box(lo, hi, ss, args.src_margin, args.min_frac, args.square) \
            or (0.0, 0.0, float(ss[0]), float(ss[1]))
        target_box = None
        if transform is not None:
            sx0, sy0, sx1, sy1 = source_box
            corners = transform(torch.tensor(
                [[sx0, sy0], [sx1, sy0], [sx0, sy1], [sx1, sy1]], dtype=torch.float32, device=DEVICE
            )).cpu().numpy()
            target_box = make_box(
                corners.min(0), corners.max(0), ts, args.tgt_margin, args.min_frac, args.square
            )
        boxes[pair_id] = {
            "affine": transform is not None,
            "source_box": source_box if target_box is not None else None,
            "target_box": target_box,
        }
    return boxes


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--backbone", default=REFERENCE_BACKBONE)
    parser.add_argument("--res", type=int, default=None, help="default: 798 (p14) / 800 (p16)")
    parser.add_argument("--src_margin", type=float, default=0.25)
    parser.add_argument("--tgt_margin", type=float, default=0.15)
    parser.add_argument("--min_frac", type=float, default=0.25)
    parser.add_argument("--square", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cache")
    args = parser.parse_args()

    start = time.time()
    queries = pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
    pairs = {
        p["pair_id"]: p
        for p in json.load(open(os.path.join(args.data_dir, f"{args.split}_pairs.json")))["pairs"]
    }
    paths = pd.concat([queries["source_image"], queries["target_image"]]).unique()
    precision = "fp32" if args.fp32 else "fp16"

    # --- crop boxes (always from the reference backbone) ---
    crops = crop_name(args)
    box_path = os.path.join(args.cache_dir, "boxes", f"{args.split}_{crops}.json")
    backbone = None
    if os.path.exists(box_path):
        boxes = json.load(open(box_path))
    else:
        ref = Backbone(REFERENCE_BACKBONE, fp16=not args.fp32)
        ref_cache = global_features(
            ref, REFERENCE_RES, paths, args.data_dir,
            os.path.join(args.cache_dir, f"{args.split}_{REFERENCE_BACKBONE}_{REFERENCE_RES}_flip_{precision}.pt"),
        )
        boxes = compute_boxes(queries, pairs, ref_cache, args)
        os.makedirs(os.path.dirname(box_path), exist_ok=True)
        json.dump(boxes, open(box_path, "w"))
        if args.backbone == REFERENCE_BACKBONE:
            backbone, global_cache = ref, ref_cache
        del ref_cache
    found = sum(b["affine"] for b in boxes.values())
    print(f"Affine found for {found}/{len(boxes)} pairs; crops = {crops}")

    # --- backbone + global features ---
    if backbone is None:
        backbone = Backbone(args.backbone, fp16=not args.fp32)
    # SD needs multiples of 64 (U-Net down/up-sampling); ViTs need multiples of their patch size
    res = args.res or (768 if backbone.kind == "sd" else 798 if backbone.patch == 14 else 800)
    if args.backbone != REFERENCE_BACKBONE or res != REFERENCE_RES or "global_cache" not in locals():
        global_cache = global_features(
            backbone, res, paths, args.data_dir,
            os.path.join(args.cache_dir, f"{args.split}_{args.backbone}_{res}_flip_{precision}.pt"),
        )

    # --- per-query similarity maps ---
    g = res // backbone.patch
    n = len(queries)
    zoom_maps = torch.zeros(n, g, g, dtype=torch.float16)
    glob_maps = torch.zeros(n, g, g, dtype=torch.float16)
    regions = torch.zeros(n, 4, dtype=torch.float32)

    for pair_id, group in tqdm(queries.groupby("pair_id", sort=False), desc=f"maps {args.backbone}"):
        meta = pairs[pair_id]
        ss, ts = tuple(meta["source_size"]), tuple(meta["target_size"])
        fs = global_cache[meta["source_image"]].to(DEVICE).float()
        ft = global_cache[meta["target_image"]].to(DEVICE).float()
        box = boxes[pair_id]

        if box["target_box"] is None:
            zs, rs, zt, rt = fs, (0, 0, *ss), ft, (0, 0, *ts)
        else:
            (zs, rs), (zt, rt) = extract_many(backbone, [
                (load_image(os.path.join(args.data_dir, meta["source_image"])), box["source_box"]),
                (load_image(os.path.join(args.data_dir, meta["target_image"])), box["target_box"]),
            ], res)

        points = torch.tensor(group[["source_x", "source_y"]].to_numpy(), dtype=torch.float32, device=DEVICE)
        q = len(points)

        local = points - torch.tensor([rs[0], rs[1]], dtype=torch.float32, device=DEVICE)
        vz = sample_source(zs, local, (rs[2], rs[3]), "bilinear")
        zoom = (vz @ zt.flatten(1)).view(q, g, g)

        vg = sample_source(fs, points, ss, "bilinear")
        full = (vg @ ft.flatten(1)).view(q, 1, *ft.shape[1:])
        centers = grid_centers(g, g, rt)
        grid = torch.stack([centers[:, 0] / ts[0] * 2 - 1, centers[:, 1] / ts[1] * 2 - 1], dim=-1)
        grid = grid.view(1, g, g, 2).expand(q, g, g, 2)
        glob = F.grid_sample(full, grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0]

        rows = torch.as_tensor(group.index.to_numpy())
        zoom_maps[rows] = zoom.half().cpu()
        glob_maps[rows] = glob.half().cpu()
        regions[rows] = torch.tensor(rt, dtype=torch.float32)

    out_path = os.path.join(
        args.cache_dir, "simmaps", f"{args.split}_{args.backbone}_{res}_{precision}_{crops}.pt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "ids": queries["id"].tolist(), "zoom": zoom_maps, "glob": glob_maps, "regions": regions,
        "backbone": args.backbone, "res": res, "crops": crops, "split": args.split,
    }, out_path)
    print(f"Saved {out_path}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
