"""
Trained head on frozen DINO crop features.

    crop -> frozen backbone features (cached by zoom_pipeline.py --save_features)
         -> small trainable head (1x1 conv -> 2x depthwise-separable 3x3 -> 1x1, + skip)
         -> L2-normalised features -> cosine similarity -> soft-argmax

Loss: cross-entropy between softmax(similarity / tau) over target patches and a Gaussian
"bump" at every accepted target (sigma = 0.05 * object scale), in both directions
(source -> target and target -> source). Augmentation: mirroring BOTH crops together.

Validation: K folds split by IMAGE (connected groups of images never cross folds), early
stopping per fold. Outputs map files compatible with evaluate_maps.py:
    cache/simmaps/train_head_<tag>.pt   out-of-fold maps for every train query (honest CV)
    cache/simmaps/test_head_<tag>.pt    test maps averaged over the K fold models

Usage
  python experiments/train_head.py \
      --features cache/features/train_dinov2_vitl14_hf_798_fp16_s0.25_t0.15_m0.25_rect.pt
"""

import os
import sys
import json
import time
import copy
import argparse
import collections

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dino_v2_param_testing import match  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_WINDOW = ("window", 2, 0.05)


# ============================================================
# MODEL
# ============================================================

class Head(nn.Module):
    def __init__(self, cin, hidden=384, out=256, dropout=0.1, arch="conv"):
        super().__init__()
        self.inp = nn.Conv2d(cin, hidden, 1)
        if arch == "conv":
            self.ctx = nn.Sequential(
                nn.GELU(), nn.Dropout2d(dropout),
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.Conv2d(hidden, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.Conv2d(hidden, hidden, 1),
            )
        else:   # per-patch only, no spatial context
            self.ctx = nn.Sequential(nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(hidden, hidden, 1))
        self.out = nn.Conv2d(hidden, out, 1)
        self.skip = nn.Conv2d(cin, out, 1, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(0.05))))

    def forward(self, x):
        h = self.inp(x)
        h = h + self.ctx(h)
        return F.normalize(self.out(F.gelu(h)) + self.skip(x), dim=1)


def apply_head(model, x, tta=True):
    """x: [B,C,g,g]. Flip-averaged head output (features are flip-equivariant, the head may not be)."""
    out = model(x)
    if tta:
        out = F.normalize(out + model(x.flip(-1)).flip(-1), dim=1)
    return out


# ============================================================
# DATA
# ============================================================

def make_folds(queries, k, seed):
    """Assign pairs to k folds so that no image appears in two folds (balanced by query count)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = queries.drop_duplicates("pair_id")
    for _, r in pairs.iterrows():
        parent[find(r.source_image)] = find(r.target_image)
    group_of_pair = {r.pair_id: find(r.source_image) for _, r in pairs.iterrows()}
    sizes = queries.groupby(queries.pair_id.map(group_of_pair)).size()
    rng = np.random.default_rng(seed)
    order = sorted(sizes.index, key=lambda g: (-sizes[g], rng.random()))
    load, fold_of_group = [0] * k, {}
    for g in order:
        f = int(np.argmin(load))
        fold_of_group[g] = f
        load[f] += sizes[g]
    return {p: fold_of_group[g] for p, g in group_of_pair.items()}, load


def merge_features(main, extra):
    """Concatenate a second feature set (e.g. SD) onto the main one, per pair/window:
    each part L2-normalised separately, extra resampled to the main grid. Crops must match."""
    out = {}
    for key, f in main.items():
        e = extra[key]
        assert tuple(e["rs"]) == tuple(f["rs"]) and tuple(e["rt"]) == tuple(f["rt"]), f"crop mismatch {key}"
        merged = {"rs": f["rs"], "rt": f["rt"]}
        for side in ("fs", "ft"):
            a = F.normalize(f[side].float(), dim=0)
            b = e[side].float()
            if b.shape[-2:] != a.shape[-2:]:
                b = F.interpolate(b[None], size=a.shape[-2:], mode="bilinear", align_corners=False)[0]
            merged[side] = torch.cat([a, F.normalize(b, dim=0)]).half()
        out[key] = merged
    return out


def build_windows(queries, features, with_targets):
    """Refinement-scale items: one per query (its own source/target windows, from
    refine_points.py --save_features). Folds still follow pair_id."""
    out = []
    for row, r in queries.iterrows():
        f = features[int(row)]
        item = {
            "pair_id": r.pair_id, "rows": np.array([row]),
            "fs": f["fs"], "ft": f["ft"], "rs": f["rs"], "rt": f["rt"],
            "src": torch.tensor([[r.source_x, r.source_y]], dtype=torch.float32),
        }
        if with_targets:
            item["tgt"] = [torch.tensor(json.loads(r.target_points), dtype=torch.float32)]
            item["scale"] = torch.tensor([r.scale], dtype=torch.float32)
        out.append(item)
    return out


def build_pairs(queries, features, with_targets):
    out = []
    for pair_id, group in queries.groupby("pair_id", sort=False):
        f = features[pair_id]
        item = {
            "pair_id": pair_id, "rows": group.index.to_numpy(),
            "fs": f["fs"], "ft": f["ft"], "rs": f["rs"], "rt": f["rt"],
            "src": torch.tensor(group[["source_x", "source_y"]].to_numpy(), dtype=torch.float32),
        }
        if with_targets:
            item["tgt"] = [torch.tensor(json.loads(t), dtype=torch.float32) for t in group.target_points]
            item["scale"] = torch.tensor(group.scale.to_numpy(), dtype=torch.float32)
        out.append(item)
    return out


def centers(g, region):
    x0, y0, w, h = region
    xs = x0 + (torch.arange(g, device=DEVICE) + 0.5) / g * w
    ys = y0 + (torch.arange(g, device=DEVICE) + 0.5) / g * h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=1)


def sample(fmap, points, region):
    """fmap [D,g,g], points [Q,2] pixels -> [Q,D] unit vectors (bilinear)."""
    x0, y0, w, h = region
    grid = torch.stack([(points[:, 0] - x0) / w * 2 - 1, (points[:, 1] - y0) / h * 2 - 1], -1)
    v = F.grid_sample(fmap[None], grid.view(1, 1, -1, 2), mode="bilinear",
                      padding_mode="border", align_corners=False)[0, :, 0].T
    return F.normalize(v, dim=-1)


def inside(points, region):
    x0, y0, w, h = region
    return (points[:, 0] >= x0) & (points[:, 0] <= x0 + w) & (points[:, 1] >= y0) & (points[:, 1] <= y0 + h)


def mirror(points, region):
    x0, _, w, _ = region
    p = points.clone()
    p[:, 0] = 2 * x0 + w - p[:, 0]
    return p


def gaussian_label(cent, targets, sigma):
    d2 = ((cent[:, None, :] - targets[None]) ** 2).sum(-1)
    lab = torch.exp(-d2 / (2 * sigma ** 2)).max(dim=1).values
    return lab / lab.sum().clamp_min(1e-8)


# ============================================================
# TRAINING
# ============================================================

def pair_loss(model, hs, ht, item, flip, reverse):
    rs, rt = item["rs"], item["rt"]
    g = ht.shape[-1]
    tau = model.log_tau.exp()
    src = item["src"].to(DEVICE)
    tgts = [t.to(DEVICE) for t in item["tgt"]]
    if flip:
        src = mirror(src, rs)
        tgts = [mirror(t, rt) for t in tgts]
    ct, cs = centers(g, rt), centers(g, rs)
    patch_t = max(rt[2], rt[3]) / g
    ratio = max(rs[2], rs[3]) / max(rt[2], rt[3])

    losses = []
    log_p = F.log_softmax((sample(hs, src, rs) @ ht.flatten(1)) / tau, dim=1)        # [Q, Nt]
    reverse_vecs, reverse_labels = [], []
    for i, t in enumerate(tgts):
        t = t[inside(t, rt)]
        if len(t) == 0:
            continue
        sigma_t = max(0.05 * float(item["scale"][i]), 0.5 * patch_t)
        losses.append(-(gaussian_label(ct, t, sigma_t) * log_p[i]).sum())
        if reverse and inside(src[i:i + 1], rs).item():
            reverse_vecs.append(sample(ht, t, rt))
            reverse_labels += [gaussian_label(cs, src[i:i + 1], sigma_t * ratio)] * len(t)
    if reverse_vecs:
        log_r = F.log_softmax((torch.cat(reverse_vecs) @ hs.flatten(1)) / tau, dim=1)
        losses.append(-(torch.stack(reverse_labels) * log_r).sum(1).mean() * 1.0)
    return torch.stack(losses).mean() if losses else None


@torch.no_grad()
def similarity_maps(models, items, tta=True, batch=8):
    """Returns {row: (sims [g,g] cpu fp16, rt)}. With several models the SIMILARITY MAPS are
    averaged (each model has its own embedding space, so features must not be averaged)."""
    out = {}
    for m in models:
        m.eval()
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        x = torch.stack([c["fs"] for c in chunk] + [c["ft"] for c in chunk]).to(DEVICE).float()
        sims_sum = [0.0] * len(chunk)
        for m in models:
            h = apply_head(m, x, tta)
            for j, c in enumerate(chunk):
                hs, ht = h[j], h[len(chunk) + j]
                g = ht.shape[-1]
                sims_sum[j] = sims_sum[j] + (sample(hs, c["src"].to(DEVICE), c["rs"]) @ ht.flatten(1)).view(-1, g, g)
        for j, c in enumerate(chunk):
            sims = sims_sum[j] / len(models)
            for q, row in enumerate(c["rows"]):
                out[int(row)] = (sims[q].half().cpu(), c["rt"])
    return out


def pck_of(maps, items, method=VAL_WINDOW):
    rows = [int(r) for c in items for r in c["rows"]]
    sims = torch.stack([maps[r][0] for r in rows]).float().to(DEVICE)
    regions = torch.tensor([maps[r][1] for r in rows], dtype=torch.float32, device=DEVICE)
    n, g, _ = sims.shape
    norm = match(sims.reshape(n, -1), g, g, (1.0, 1.0), method)
    pred = (regions[:, :2] + norm * regions[:, 2:]).cpu().numpy()
    tg = {int(r): t.numpy() for c in items for r, t in zip(c["rows"], c["tgt"])}
    sc = {int(r): float(s) for c in items for r, s in zip(c["rows"], c["scale"])}
    e = np.array([np.linalg.norm(tg[r] - p, axis=1).min() / sc[r] for r, p in zip(rows, pred)])
    return float((e < 0.10).mean())


def train_fold(train_items, val_items, cin, args, fold, fixed_epochs=None):
    """Early stopping on val_items, or (fixed_epochs) train exactly that many epochs without
    looking at any validation data. Returns (model, best_val, best_epoch)."""
    torch.manual_seed(args.seed + fold)
    model = Head(cin, args.hidden, args.out, args.dropout, args.arch).to(DEVICE)
    decay = [p for n, p in model.named_parameters() if n != "log_tau"]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": [model.log_tau], "weight_decay": 0.0}], lr=args.lr)
    rng = np.random.default_rng(args.seed + fold)
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    model.history = []
    for epoch in range(fixed_epochs or args.epochs):
        model.train()
        order = rng.permutation(len(train_items))
        total, steps = 0.0, 0
        for k in range(0, len(order), args.batch):
            chunk = [train_items[i] for i in order[k:k + args.batch]]
            flips = [bool(rng.random() < 0.5) and not args.no_flip for _ in chunk]
            fs = [c["fs"].flip(-1) if f else c["fs"] for c, f in zip(chunk, flips)]
            ft = [c["ft"].flip(-1) if f else c["ft"] for c, f in zip(chunk, flips)]
            h = model(torch.stack(fs + ft).to(DEVICE).float())
            losses = [pair_loss(model, h[j], h[len(chunk) + j], c, f, not args.no_reverse)
                      for j, (c, f) in enumerate(zip(chunk, flips))]
            losses = [l for l in losses if l is not None]
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        if fixed_epochs:
            print(f"  fold {fold} final-fit epoch {epoch:2d} loss {total / max(steps, 1):.3f}", flush=True)
            continue
        val = pck_of(similarity_maps([model], val_items, tta=True), val_items)
        model.history.append(val)
        if val > best:
            best, best_state, best_epoch, bad = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
        print(f"  fold {fold} epoch {epoch:2d} loss {total / max(steps, 1):.3f} "
              f"tau {model.log_tau.exp().item():.3f} val PCK@0.10 {val:.4f}{' *' if bad == 0 else ''}", flush=True)
        if bad >= args.patience:
            break
    if not fixed_epochs:
        model.load_state_dict(best_state)
    return model, best, best_epoch


# ============================================================
# MAIN
# ============================================================

def save_maps(path, ids, maps, split, tag, prior_path=None):
    """glob = copy of the head map, or (windows) the coarse prior from the refine map file,
    so evaluate_maps.py's alpha blends the head with the coarse prior like the raw windows."""
    rows = list(range(len(ids)))
    g = maps[rows[0]][0].shape[-1]
    zoom = torch.stack([maps[r][0] for r in rows])
    regions = torch.tensor([maps[r][1] for r in rows], dtype=torch.float32)
    glob = zoom.clone()
    if prior_path:
        prior = torch.load(prior_path)
        assert prior["ids"] == ids, "prior map file is in a different query order"
        assert torch.allclose(prior["regions"], regions), "prior windows differ from feature windows"
        glob = prior["glob"]
    torch.save({"ids": ids, "zoom": zoom, "glob": glob, "regions": regions,
                "backbone": f"head_{tag}", "res": g, "crops": "head", "split": split}, path)
    print("Saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="train feature file (test_* derived from it)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--arch", choices=["conv", "linear"], default="conv")
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--out", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_reverse", action="store_true")
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--nested", action="store_true",
                        help="choose epochs on an inner fold; report the outer fold untouched (honest CV)")
    parser.add_argument("--select", choices=["best", "one_se", "fixed"], default="best",
                        help="nested epoch rule: best inner epoch / earliest within 1 SE / fixed count")
    parser.add_argument("--fixed_epochs", type=int, default=2)
    parser.add_argument("--extra_features", nargs="*", default=[],
                        help="train feature files concatenated onto --features (test_* derived)")
    parser.add_argument("--windows", action="store_true",
                        help="refinement-scale training on per-query windows (refine_points --save_features)")
    parser.add_argument("--prior_maps", default=None,
                        help="(windows) train refine map file whose coarse prior is copied into the output")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    tag = args.tag or f"{args.arch}_h{args.hidden}_o{args.out}" + ("_norev" if args.no_reverse else "") \
        + ("_noflip" if args.no_flip else "") + ("_nested" if args.nested else "") \
        + ({"best": "", "one_se": "_1se", "fixed": f"_fix{args.fixed_epochs}"}[args.select]) \
        + ("_win" if args.windows else "") + ("_fused" if args.extra_features else "")
    build = build_windows if args.windows else build_pairs

    def load_features(path):
        feats = torch.load(path)
        for extra in args.extra_features:
            extra_path = extra if "train_" in os.path.basename(path) else os.path.join(
                os.path.dirname(extra), os.path.basename(extra).replace("train_", "test_", 1))
            feats = merge_features(feats, torch.load(extra_path))
        return feats
    start = time.time()

    queries = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    features = load_features(args.features)
    items = build(queries, features, with_targets=True)
    cin = items[0]["fs"].shape[0]
    fold_of_pair, load = make_folds(queries, args.folds, args.seed)
    print(f"{len(items)} pairs, feature dim {cin}, fold sizes (queries): {load}")

    # raw-feature baseline on the same evaluation (no head)
    raw = {}
    for c in items:
        fs, ft = c["fs"].to(DEVICE).float(), c["ft"].to(DEVICE).float()
        g = ft.shape[-1]
        sims = (sample(fs, c["src"].to(DEVICE), c["rs"]) @ F.normalize(ft, dim=0).flatten(1)).view(-1, g, g)
        for q, row in enumerate(c["rows"]):
            raw[int(row)] = (sims[q].half().cpu(), c["rt"])
    print(f"raw features (no head) PCK@0.10 = {pck_of(raw, items):.4f}")

    oof, models, fold_scores = {}, [], []
    for fold in range(args.folds):
        tr = [c for c in items if fold_of_pair[c["pair_id"]] != fold]
        va = [c for c in items if fold_of_pair[c["pair_id"]] == fold]
        if args.nested:
            # 1) choose the number of epochs on an INNER fold (outer fold untouched)
            inner = (fold + 1) % args.folds
            tr_in = [c for c in tr if fold_of_pair[c["pair_id"]] != inner]
            va_in = [c for c in tr if fold_of_pair[c["pair_id"]] == inner]
            if args.select == "fixed":
                inner_best, best_epoch = float("nan"), args.fixed_epochs - 1
            else:
                sel_model, inner_best, best_epoch = train_fold(tr_in, va_in, cin, args, fold)
                if args.select == "one_se":
                    # earliest epoch within one standard error of the inner best
                    n_inner = sum(len(c["rows"]) for c in va_in)
                    se = np.sqrt(inner_best * (1 - inner_best) / n_inner)
                    best_epoch = next(e for e, v in enumerate(sel_model.history) if v >= inner_best - se)
            # 2) retrain on all training folds for exactly that many epochs, 3) score outer fold once
            model, _, _ = train_fold(tr, None, cin, args, fold, fixed_epochs=best_epoch + 1)
            maps = similarity_maps([model], va)
            score = pck_of(maps, va)
            print(f"fold {fold}: epochs chosen on inner fold {inner}: {best_epoch + 1} "
                  f"(inner val {inner_best:.4f}) -> OUTER PCK@0.10 {score:.4f}  "
                  f"(raw on same fold {pck_of(raw, va):.4f})  [{time.time() - start:.0f}s]", flush=True)
        else:
            model, score, _ = train_fold(tr, va, cin, args, fold)
            maps = similarity_maps([model], va)
            print(f"fold {fold}: best val PCK@0.10 {score:.4f}  "
                  f"(raw on same fold {pck_of(raw, va):.4f})  [{time.time() - start:.0f}s]", flush=True)
        models.append(model)
        fold_scores.append(score)
        oof.update(maps)

    print(f"\nOUT-OF-FOLD head PCK@0.10 = {pck_of(oof, items):.4f}   vs raw {pck_of(raw, items):.4f}")
    os.makedirs(os.path.join(args.cache_dir, "simmaps"), exist_ok=True)
    os.makedirs(os.path.join(args.cache_dir, "heads"), exist_ok=True)
    save_maps(os.path.join(args.cache_dir, "simmaps", f"train_head_{tag}.pt"), queries["id"].tolist(), oof,
              "train", tag, prior_path=args.prior_maps)
    torch.save({"args": vars(args), "states": [m.state_dict() for m in models], "cin": cin},
               os.path.join(args.cache_dir, "heads", f"head_{tag}.pt"))

    test_features_path = os.path.join(os.path.dirname(args.features),
                                      os.path.basename(args.features).replace("train_", "test_", 1))
    if os.path.exists(test_features_path):
        test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        test_items = build(test, load_features(test_features_path), with_targets=False)
        test_prior = None
        if args.prior_maps:
            test_prior = os.path.join(os.path.dirname(args.prior_maps),
                                      os.path.basename(args.prior_maps).replace("train_", "test_", 1))
        save_maps(os.path.join(args.cache_dir, "simmaps", f"test_head_{tag}.pt"), test["id"].tolist(),
                  similarity_maps(models, test_items), "test", tag, prior_path=test_prior)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
