"""
Render sheets of training failures of the current best model for manual inspection.

Per panel:  SOURCE  cyan ring = query point, white dots = the pair's other source keypoints (numbered)
            TARGET  green dot + circle = accepted target(s) with the 10% tolerance radius,
                    red X = final prediction, orange + = coarse prediction,
                    yellow box = object crop, magenta box = refinement window
Writes experiments/inspection/sheet_XX.jpg and inspection_queries.csv (with an empty `tag` column).
"""
import os, sys, json
import numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_maps import load_maps, blend, predict, target_array, errors

OUT = "experiments/inspection"
CROPS = "s0.25_t0.15_m0.25_rect"
q = pd.read_csv("data/train.csv"); ids = q.id.tolist()
pairs = {p["pair_id"]: p for p in json.load(open("data/train_pairs.json"))["pairs"]}
boxes = json.load(open(f"cache/boxes/train_{CROPS}.json"))
final = pd.read_csv("experiments/results/maps_step4b_refineBL_best_per_query.csv").set_index("id").loc[ids]
z, g, r = load_maps([f"cache/simmaps/train_dinov2_vitb14_798_fp16_{CROPS}.pt",
                     f"cache/simmaps/train_dinov2_vitl14_hf_798_fp16_{CROPS}.pt"], ids)
coarse = predict(blend(z, g, [0.75, 1.0], 0.0), r, ("window", 2, 0.08))
windows = torch.load("cache/simmaps/train_refine_dinov2_vitb14_448_f0.4_coarse_BL.pt")["regions"].numpy()

T = [np.asarray(json.loads(s)) for s in q.target_points]
d = q.copy()
d["px"], d["py"], d["err"] = final.x.values, final.y.values, final.error.values
d["cx"], d["cy"] = coarse[:, 0], coarse[:, 1]
d["rel"] = d.scale / d.pair_id.map(lambda p: max(pairs[p]["target_size"]))
cat = []
for i, row in d.iterrows():
    tb = boxes[row.pair_id]["target_box"]
    in_crop = tb is None or any(tb[0] <= x <= tb[2] and tb[1] <= y <= tb[3] for x, y in T[i])
    p = np.array([row.px, row.py])
    others = d.index[(d.pair_id == row.pair_id) & (d.id != row.id)]
    other_kp = any(np.linalg.norm(T[j] - p, axis=1).min() / row.scale < 0.1 for j in others)
    cat.append("correct" if row.err < 0.1 else "crop_missed" if not in_crop else "near_miss" if row.err < 0.2
               else "other_keypoint" if other_kp else "far")
d["auto"] = cat
print("auto categories:", d.auto.value_counts().to_dict())

# ---- stratified sample of failures ----
rng = np.random.default_rng(7)
wrong = d[d.auto != "correct"]
pick = []
zero_pairs = d.groupby("pair_id").err.apply(lambda e: (e < 0.1).sum() == 0)
for pid in zero_pairs[zero_pairs].index[:16]:
    pick.append(rng.choice(wrong.index[wrong.pair_id == pid]))
d["size_bin"] = pd.cut(d.rel, [0, .2, .4, .7, 10], labels=["small", "medium", "large", "xlarge"])
for b in ["small", "medium", "large", "xlarge"]:
    pool = [i for i in wrong.index[(wrong.auto == "far") & (d.size_bin[wrong.index] == b)] if i not in pick]
    pick += list(rng.choice(pool, size=min(4, len(pool)), replace=False))
for c, k in [("other_keypoint", 8), ("near_miss", 8)]:
    pool = [i for i in wrong.index[wrong.auto == c] if i not in pick]
    pick += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
sel = d.loc[pick].copy()
sel["zero_pair"] = sel.pair_id.map(zero_pairs)
print("zero-correct pairs:", int(zero_pairs.sum()), "| sampled failures:", len(sel))

# ---- render ----
H = 300
def panel(n, row):
    meta = pairs[row.pair_id]
    s = Image.open("data/" + row.source_image).convert("RGB"); t = Image.open("data/" + row.target_image).convert("RGB")
    ks, kt = H / s.height, H / t.height
    s = s.resize((max(1, int(s.width * ks)), H)); t = t.resize((max(1, int(t.width * kt)), H))
    ds, dt = ImageDraw.Draw(s), ImageDraw.Draw(t)
    for j, (x, y) in enumerate(meta["source_xy"]):
        x, y = x * ks, y * ks
        ds.ellipse([x - 3, y - 3, x + 3, y + 3], fill="white", outline="black"); ds.text((x + 4, y - 6), str(j), fill="white")
    x, y = row.source_x * ks, row.source_y * ks
    ds.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(0, 255, 255), width=3)
    tb = boxes[row.pair_id]["target_box"]
    if tb: dt.rectangle([v * kt for v in tb], outline=(255, 220, 0), width=2)
    w = windows[row.name]; dt.rectangle([w[0] * kt, w[1] * kt, (w[0] + w[2]) * kt, (w[1] + w[3]) * kt], outline=(255, 0, 255), width=2)
    rad = 0.1 * row.scale * kt
    for gx, gy in T[row.name]:
        gx, gy = gx * kt, gy * kt
        dt.ellipse([gx - rad, gy - rad, gx + rad, gy + rad], outline=(0, 255, 0), width=2); dt.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=(0, 255, 0))
    px, py = row.cx * kt, row.cy * kt
    dt.line([px - 7, py, px + 7, py], fill=(255, 140, 0), width=3); dt.line([px, py - 7, px, py + 7], fill=(255, 140, 0), width=3)
    px, py = row.px * kt, row.py * kt
    dt.line([px - 9, py - 9, px + 9, py + 9], fill=(255, 0, 0), width=4); dt.line([px - 9, py + 9, px + 9, py - 9], fill=(255, 0, 0), width=4)
    img = Image.new("RGB", (s.width + t.width + 10, H + 22), "white")
    img.paste(s, (0, 22)); img.paste(t, (s.width + 10, 22))
    ImageDraw.Draw(img).text((4, 4), f"#{n}  {row.id}  err={row.err:.2f}  size={row.rel:.2f}  auto={row.auto}"
                             f"{'  ZERO-CORRECT PAIR' if row.zero_pair else ''}  ntgt={len(T[row.name])}", fill="black")
    return img
sel = sel.reset_index().rename(columns={"index": "row"}).set_index("row", drop=False)
sel.index.name = None
sel["panel"] = range(1, len(sel) + 1)
for k in range(0, len(sel), 4):
    chunk = sel.iloc[k:k + 4]
    ims = [panel(int(p), rw.rename(int(rw.row))) for p, (_, rw) in zip(chunk.panel, chunk.iterrows())]
    W = max(i.width for i in ims); S = Image.new("RGB", (W, sum(i.height + 6 for i in ims)), "gray")
    yy = 0
    for im in ims: S.paste(im, (0, yy)); yy += im.height + 6
    S.save(f"{OUT}/sheet_{k // 4 + 1:02d}.jpg", quality=85)
sel["tag"] = ""
sel[["panel", "id", "pair_id", "err", "rel", "size_bin", "auto", "zero_pair", "tag"]].to_csv(f"{OUT}/inspection_queries.csv", index=False)
print("sheets:", (len(sel) + 3) // 4)
