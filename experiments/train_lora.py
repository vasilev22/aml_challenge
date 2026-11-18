"""
Step 3: LoRA fine-tuning of the DINOv2 backbone on the zoom crops.

Unlike the head (small conv on FROZEN features), LoRA adds small trainable low-rank matrices
inside every attention layer, so the backbone itself adapts and can use global context
(which way the object faces, which end is the front).

    crop pair (same crop boxes as zoom_pipeline.py) -> augment -> DINOv2 + LoRA
        -> L2-normalised patch features -> similarity -> loss (same as train_head.py:
           Gaussian bump at accepted targets, both directions)

Augmentation (training only): crop-box jitter (scale/shift), colour jitter per image,
mirroring BOTH crops together.

Modes
  --mode dev : train on all folds except --dev_fold, report val PCK every epoch
               (used once to pick the number of epochs)
  --mode cv  : K-fold, fixed --epochs, out-of-fold maps for train + fold-averaged test maps
               (files compatible with evaluate_maps.py; same crop regions as the other maps)

Usage
  python experiments/train_lora.py --mode dev --epochs 12
  python experiments/train_lora.py --mode cv  --epochs 6
"""

import os
import sys
import json
import time
import math
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoom_pipeline import Backbone, load_image, preprocess  # noqa: E402
from train_head import make_folds, pair_loss, sample, pck_of  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LoRA
# ============================================================

class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))      # starts as identity
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scale


class LoRAModel(nn.Module):
    """Frozen DINOv2 (torch hub) + LoRA on attention qkv/proj (+ optional MLP)."""

    def __init__(self, backbone_name, rank, alpha, mlp=False):
        super().__init__()
        bb = Backbone(backbone_name, fp16=True)                 # loads + fused-attention patch
        self.vit = bb.model
        self.patch = bb.patch
        self.mean, self.std = bb.mean, bb.std
        for p in self.vit.parameters():
            p.requires_grad = False
        for block in self.vit.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
            block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)
            if mlp:
                block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, alpha)
                block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, alpha)
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(0.05))))

    def features(self, x):
        """x [B,3,H,W] normalised -> [B,C,h,w] L2-normalised."""
        b, _, height, width = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            tokens = self.vit.forward_features(x)["x_norm_patchtokens"]
        f = tokens.float().transpose(1, 2).reshape(b, -1, height // self.patch, width // self.patch)
        return F.normalize(f, dim=1)

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


# ============================================================
# DATA
# ============================================================

def build_items(queries, pairs, boxes, images, with_targets):
    items = []
    for pair_id, group in queries.groupby("pair_id", sort=False):
        meta, box = pairs[pair_id], boxes[pair_id]
        item = {
            "pair_id": pair_id, "rows": group.index.to_numpy(),
            "simg": images[meta["source_image"]], "timg": images[meta["target_image"]],
            "sbox": box["source_box"], "tbox": box["target_box"],          # None -> full image
            "src": torch.tensor(group[["source_x", "source_y"]].to_numpy(), dtype=torch.float32),
        }
        if with_targets:
            item["tgt"] = [torch.tensor(json.loads(t), dtype=torch.float32) for t in group.target_points]
            item["scale"] = torch.tensor(group.scale.to_numpy(), dtype=torch.float32)
        items.append(item)
    return items


def jitter_box(box, size, rng, amount=0.08):
    width, height = size
    x0, y0, x1, y1 = box if box is not None else (0.0, 0.0, float(width), float(height))
    w, h = x1 - x0, y1 - y0
    s = 1 + rng.uniform(-amount, amount)
    cx = (x0 + x1) / 2 + rng.uniform(-amount, amount) * w
    cy = (y0 + y1) / 2 + rng.uniform(-amount, amount) * h
    w, h = min(w * s, width), min(h * s, height)
    x0 = min(max(cx - w / 2, 0.0), width - w)
    y0 = min(max(cy - h / 2, 0.0), height - h)
    return (x0, y0, x0 + w, y0 + h)


def colour_jitter(x, rng, amount=0.2):
    x = TF.adjust_brightness(x, 1 + rng.uniform(-amount, amount))
    x = TF.adjust_contrast(x, 1 + rng.uniform(-amount, amount))
    return TF.adjust_saturation(x, 1 + rng.uniform(-amount, amount))


def mirror(points, region):
    x0, _, w, _ = region
    p = points.clone()
    p[:, 0] = 2 * x0 + w - p[:, 0]
    return p


def make_view(item, res, model, rng=None):
    """Returns (source tensor, target tensor, training item with coords in the same frame)."""
    simg, timg = item["simg"], item["timg"]
    sbox, tbox = item["sbox"], item["tbox"]
    if rng is not None:
        sbox = jitter_box(sbox, (simg.shape[2], simg.shape[1]), rng)
        tbox = jitter_box(tbox, (timg.shape[2], timg.shape[1]), rng)
        simg, timg = colour_jitter(simg, rng), colour_jitter(timg, rng)
    xs, rs = preprocess(simg, sbox, res, model.mean, model.std)
    xt, rt = preprocess(timg, tbox, res, model.mean, model.std)
    view = {"rs": rs, "rt": rt, "src": item["src"]}
    if "tgt" in item:
        view["tgt"], view["scale"] = item["tgt"], item["scale"]
    if rng is not None and rng.random() < 0.5:                     # mirror both crops together
        xs, xt = xs.flip(-1), xt.flip(-1)
        view["src"] = mirror(item["src"], rs)
        view["tgt"] = [mirror(t, rt) for t in item["tgt"]]
    return xs, xt, view


# ============================================================
# TRAIN / EVAL
# ============================================================

@torch.no_grad()
def lora_maps(models, items, res):
    """{row: (sims [g,g] fp16 cpu, region)} with flip TTA, similarity maps averaged over models."""
    out = {}
    for m in models:
        m.eval()
    for item in items:
        xs, xt, view = make_view(item, res, models[0])
        x = torch.stack([xs, xt]).to(DEVICE)
        sims = 0
        for m in models:
            f = m.features(x)
            f = F.normalize(f + m.features(x.flip(-1)).flip(-1), dim=1)
            g = f.shape[-1]
            sims = sims + (sample(f[0], view["src"].to(DEVICE), view["rs"]) @ f[1].flatten(1)).view(-1, g, g)
        sims = sims / len(models)
        for q, row in enumerate(item["rows"]):
            out[int(row)] = (sims[q].half().cpu(), view["rt"])
    return out


def train(train_items, val_items, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LoRAModel(args.backbone, args.rank, args.alpha, args.mlp).to(DEVICE)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and n != "log_tau"]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": [model.log_tau], "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = math.ceil(len(train_items) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda")
    n_params = sum(p.numel() for p in model.trainable())
    print(f"  trainable parameters: {n_params / 1e6:.2f}M", flush=True)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_items))
        total, steps, t0 = 0.0, 0, time.time()
        for k in range(0, len(order), args.batch):
            chunk = [train_items[i] for i in order[k:k + args.batch]]
            views, xs = [], []
            for item in chunk:
                s, t, v = make_view(item, args.res, model, rng)
                xs += [s, t]
                views.append(v)
            f = model.features(torch.stack(xs).to(DEVICE))
            losses = [pair_loss(model, f[2 * j], f[2 * j + 1], v, False, True) for j, v in enumerate(views)]
            losses = [l for l in losses if l is not None]
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            steps += 1
        msg = f"  epoch {epoch:2d} loss {total / max(steps, 1):.3f} ({time.time() - t0:.0f}s)"
        if val_items is not None:
            val = pck_of(lora_maps([model], val_items, args.res), val_items)
            history.append(val)
            msg += f"  val PCK@0.10 {val:.4f}"
        print(msg, flush=True)
    return model, history


def save_maps(path, ids, maps, split, tag):
    rows = list(range(len(ids)))
    zoom = torch.stack([maps[r][0] for r in rows])
    regions = torch.tensor([maps[r][1] for r in rows], dtype=torch.float32)
    torch.save({"ids": ids, "zoom": zoom, "glob": zoom.clone(), "regions": regions,
                "backbone": f"lora_{tag}", "res": zoom.shape[-1], "crops": "lora", "split": split}, path)
    print("Saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "cv"], default="dev")
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--res", type=int, default=644)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--mlp", action="store_true", help="LoRA on the MLP layers too")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4, help="pairs per step")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dev_fold", type=int, default=1)
    parser.add_argument("--crops", default="s0.25_t0.15_m0.25_rect")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cache")
    args = parser.parse_args()
    tag = f"{args.backbone}_r{args.rank}{'_mlp' if args.mlp else ''}_res{args.res}_e{args.epochs}"
    start = time.time()

    def load_split(split):
        queries = pd.read_csv(os.path.join(args.data_dir, f"{split}.csv"))
        pairs = {p["pair_id"]: p for p in
                 json.load(open(os.path.join(args.data_dir, f"{split}_pairs.json")))["pairs"]}
        boxes = json.load(open(os.path.join(args.cache_dir, "boxes", f"{split}_{args.crops}.json")))
        paths = pd.concat([queries.source_image, queries.target_image]).unique()
        images = {p: load_image(os.path.join(args.data_dir, p)) for p in paths}
        return queries, build_items(queries, pairs, boxes, images, with_targets=(split == "train"))

    queries, items = load_split("train")
    fold_of_pair, load = make_folds(queries, args.folds, 0)                  # same folds as the heads
    print(f"{len(items)} pairs, fold sizes {load}, tag {tag}", flush=True)

    if args.mode == "dev":
        tr = [c for c in items if fold_of_pair[c["pair_id"]] != args.dev_fold]
        va = [c for c in items if fold_of_pair[c["pair_id"]] == args.dev_fold]
        print(f"dev: train {len(tr)} pairs, validate on fold {args.dev_fold} ({len(va)} pairs)", flush=True)
        # baseline: the frozen backbone (LoRA B=0 -> identical to DINOv2) on the same fold
        base = LoRAModel(args.backbone, args.rank, args.alpha, args.mlp).to(DEVICE)
        print(f"  frozen backbone val PCK@0.10 {pck_of(lora_maps([base], va, args.res), va):.4f}", flush=True)
        del base
        _, history = train(tr, va, args, seed=0)
        best = int(np.argmax(history))
        print(f"DEV best epoch {best + 1} PCK {history[best]:.4f}  [{time.time() - start:.0f}s]")
        return

    oof, models = {}, []
    for fold in range(args.folds):
        tr = [c for c in items if fold_of_pair[c["pair_id"]] != fold]
        va = [c for c in items if fold_of_pair[c["pair_id"]] == fold]
        model, _ = train(tr, None, args, seed=fold)
        maps = lora_maps([model], va, args.res)
        oof.update(maps)
        print(f"fold {fold}: PCK@0.10 {pck_of(maps, va):.4f}  [{time.time() - start:.0f}s]", flush=True)
        model.cpu()
        models.append(model)
        torch.cuda.empty_cache()
    print(f"OUT-OF-FOLD LoRA PCK@0.10 = {pck_of(oof, items):.4f}")
    save_maps(os.path.join(args.cache_dir, "simmaps", f"train_lora_{tag}.pt"), queries["id"].tolist(), oof, "train", tag)

    test_queries, test_items = load_split("test")
    test_maps = {}
    for m in models:                                    # one model on the GPU at a time
        m.to(DEVICE)
        maps = lora_maps([m], test_items, args.res)
        for r, (s, region) in maps.items():
            prev = test_maps.get(r)
            test_maps[r] = (s.float() if prev is None else prev[0] + s.float(), region)
        m.cpu()
    test_maps = {r: ((s / len(models)).half(), region) for r, (s, region) in test_maps.items()}
    save_maps(os.path.join(args.cache_dir, "simmaps", f"test_lora_{tag}.pt"), test_queries["id"].tolist(),
              test_maps, "test", tag)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
