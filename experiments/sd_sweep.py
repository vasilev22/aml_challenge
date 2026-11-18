"""
Stable Diffusion feature sweep on the zoom crops (one extraction run -> many variants).

Per crop pair, SD 2.1 runs on [source, source-flipped, target, target-flipped] for N noise
draws at one timestep; decoder blocks 1 (stride 16, 1280 ch) and 2 (stride 8, 640 ch) are
captured in the same pass. Running averages are snapshotted after 2/4/8 draws, and every
snapshot is turned into maps with and without flip averaging:

    layers {up1, up2} x ensembles {2, 4, 8} x flip {on, off}  ->  12 map files

Map files match evaluate_maps.py (same crop boxes/regions as the DINOv2 maps), so layer
combinations and blends with the current pipeline are evaluated there.

Usage
  python experiments/sd_sweep.py --split train --timestep 261
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
from zoom_pipeline import load_image, preprocess, SD_REPOS  # noqa: E402
from dino_v2_param_testing import sample_source  # noqa: E402

DEVICE = torch.device("cuda")
RES = 768
SNAPSHOTS = [2, 4, 8]


class Stop(Exception):
    pass


def load_sd(name, prompt=""):
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_REPOS[name], torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
        safety_checker=None, requires_safety_checker=False)
    tokens = pipe.tokenizer([prompt], padding="max_length", return_tensors="pt",
                            max_length=pipe.tokenizer.model_max_length)
    with torch.inference_mode():
        text = pipe.text_encoder.to(DEVICE)(tokens.input_ids.to(DEVICE))[0]
    vae, unet = pipe.vae.to(DEVICE).eval(), pipe.unet.to(DEVICE).eval()
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    captured = {}

    def hook1(m, i, o):
        captured["up1"] = o

    def hook2(m, i, o):
        captured["up2"] = o
        raise Stop

    unet.up_blocks[1].register_forward_hook(hook1)
    unet.up_blocks[2].register_forward_hook(hook2)
    return vae, unet, sched, text, captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--sd", default="sd21")
    parser.add_argument("--timestep", type=int, default=261)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--crops", default="s0.25_t0.15_m0.25_rect")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--only", default=None, help="save only this variant, e.g. up1_e8_flip (for test)")
    parser.add_argument("--res", type=int, default=768, help="multiple of 64; SD 1.5 is native at 512")
    parser.add_argument("--prompt", default="", help='e.g. "a photo of an object"')
    parser.add_argument("--save_features", default=None,
                        help="also save per-pair crop features of this variant (e.g. up1_e8_flip) for train_head.py")
    args = parser.parse_args()
    start = time.time()
    global RES
    RES = args.res

    queries = pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
    pairs = {p["pair_id"]: p for p in json.load(open(os.path.join(args.data_dir, f"{args.split}_pairs.json")))["pairs"]}
    boxes = json.load(open(os.path.join(args.cache_dir, "boxes", f"{args.split}_{args.crops}.json")))
    vae, unet, sched, text, captured = load_sd(args.sd, args.prompt)
    t = torch.tensor([args.timestep], device=DEVICE)

    n = len(queries)
    grids = {"up1": RES // 16, "up2": RES // 8}
    variants = [(layer, e, flip) for layer in grids for e in SNAPSHOTS if e <= args.samples for flip in (True, False)]
    name = lambda v: f"{v[0]}_e{v[1]}_{'flip' if v[2] else 'noflip'}"
    if args.only:
        variants = [v for v in variants if name(v) == args.only]
    maps = {v: torch.zeros(n, grids[v[0]], grids[v[0]], dtype=torch.float16) for v in variants}
    regions = torch.zeros(n, 4)
    saved_features = {}

    for pair_id, group in tqdm(queries.groupby("pair_id", sort=False), desc=f"SD t={args.timestep}"):
        meta, box = pairs[pair_id], boxes[pair_id]
        simg = load_image(os.path.join(args.data_dir, meta["source_image"]))
        timg = load_image(os.path.join(args.data_dir, meta["target_image"]))
        xs, rs = preprocess(simg, box["source_box"], RES, [0.5] * 3, [0.5] * 3)
        xt, rt = preprocess(timg, box["target_box"], RES, [0.5] * 3, [0.5] * 3)
        x = torch.stack([xs, xs.flip(-1), xt, xt.flip(-1)]).to(DEVICE).half()
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        with torch.inference_mode():
            lat = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            sums = {"up1": 0, "up2": 0}
            snaps = {}
            for k in range(args.samples):
                noise = torch.randn(lat.shape, generator=gen, device=DEVICE, dtype=lat.dtype)
                try:
                    unet(sched.add_noise(lat, noise, t), t, encoder_hidden_states=text.expand(4, -1, -1))
                except Stop:
                    pass
                for layer in sums:
                    sums[layer] = sums[layer] + captured[layer].float()
                if k + 1 in SNAPSHOTS:
                    snaps[k + 1] = {layer: sums[layer] / (k + 1) for layer in sums}

            points = torch.tensor(group[["source_x", "source_y"]].to_numpy(), dtype=torch.float32, device=DEVICE)
            local = points - torch.tensor([rs[0], rs[1]], dtype=torch.float32, device=DEVICE)
            q = len(points)
            for v in variants:
                layer, e, flip = v
                f = snaps[e][layer]
                fs, ft = F.normalize(f[0], dim=0), F.normalize(f[2], dim=0)
                if flip:
                    fs = F.normalize(fs + F.normalize(f[1].flip(-1), dim=0), dim=0)
                    ft = F.normalize(ft + F.normalize(f[3].flip(-1), dim=0), dim=0)
                g = ft.shape[-1]
                if args.save_features == name(v):
                    saved_features[pair_id] = {"fs": fs.half().cpu(), "rs": tuple(rs),
                                               "ft": ft.half().cpu(), "rt": tuple(rt)}
                vec = sample_source(fs, local, (rs[2], rs[3]), "bilinear")
                maps[v][torch.as_tensor(group.index.to_numpy())] = (vec @ ft.flatten(1)).view(q, g, g).half().cpu()
        regions[torch.as_tensor(group.index.to_numpy())] = torch.tensor(rt, dtype=torch.float32)

    os.makedirs(os.path.join(args.cache_dir, "simmaps"), exist_ok=True)
    for v in variants:
        prefix = ("sdx" if args.sd == "sd21" else f"sdx-{args.sd}") \
            + (f"-r{args.res}" if args.res != 768 else "") + ("-prompt" if args.prompt else "")
        path = os.path.join(args.cache_dir, "simmaps", f"{args.split}_{prefix}_{name(v)}_t{args.timestep}_{args.crops}.pt")
        torch.save({"ids": queries["id"].tolist(), "zoom": maps[v], "glob": maps[v].clone(), "regions": regions,
                    "backbone": f"sd_{name(v)}", "res": RES, "crops": args.crops, "split": args.split}, path)
    print(f"Saved {len(variants)} variants  ({time.time() - start:.0f}s)")
    if args.save_features:
        prefix = ("sdx" if args.sd == "sd21" else f"sdx-{args.sd}") \
            + (f"-r{args.res}" if args.res != 768 else "") + ("-prompt" if args.prompt else "")
        path = os.path.join(args.cache_dir, "features",
                            f"{args.split}_{prefix}_{args.save_features}_t{args.timestep}_{args.crops}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(saved_features, path)
        print("Saved", path)


if __name__ == "__main__":
    main()
