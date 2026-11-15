"""
Baseline: DINOv3 dense features + nearest neighbor (cosine) per semantic correspondence.

Uso:
  # valutazione su un sottoinsieme del train
    python test_dinov3.py --root . --mode eval --n_pairs 60

    # predizioni sul test -> predictions.csv (id,x,y)
    python test_dinov3.py --root . --mode test

    # conversione nel formato di submission
    python make_submission.py --predictions predictions.csv --output submission.csv

Requisiti: torch, transformers>=4.56, pillow, pandas, numpy
Il modello DINOv3 su HuggingFace e' "gated": serve accettare la licenza e fare `huggingface-cli login`.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FeatureExtractor:
    def __init__(self, name="facebook/dinov3-vitb16-pretrain-lvd1689m", size=768, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.patch = self.model.config.patch_size           # 16
        self.n_reg = getattr(self.model.config, "num_register_tokens", 0)
        self.size = size                                     # lato (multiplo di 16) a cui ridimensiono
        self.cache = {}

    @torch.no_grad()
    def __call__(self, path):
        """Ritorna feature map normalizzata [1, D, h, w] (immagine stirata a size x size)."""
        if path in self.cache:
            return self.cache[path]
        img = Image.open(path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255)[None]
        x = ((x - MEAN) / STD).to(self.device)
        out = self.model(pixel_values=x).last_hidden_state   # [1, 1+n_reg+N, D]
        tokens = out[:, 1 + self.n_reg:]
        h = w = self.size // self.patch
        feat = tokens.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        feat = F.normalize(feat, dim=1)
        if len(self.cache) > 200:
            self.cache.clear()
        self.cache[path] = feat
        return feat


def sample_feature(feat, x, y, W, H):
    """Feature bilineare nel punto (x,y) in pixel originali."""
    grid = torch.tensor([[[[x / W * 2 - 1, y / H * 2 - 1]]]], dtype=feat.dtype, device=feat.device)
    f = F.grid_sample(feat, grid, mode="bilinear", align_corners=False)  # [1,D,1,1]
    return F.normalize(f.flatten(1), dim=1)                               # [1,D]


def match(feat_t, f_src, W_t, H_t, up=4):
    """Argmax della similarita' coseno sulla mappa target (upsampled) -> (x,y) pixel originali."""
    sim = torch.einsum("bd,bdhw->bhw", f_src, feat_t)[None]               # [1,1,h,w]
    sim = F.interpolate(sim, scale_factor=up, mode="bicubic", align_corners=False)
    hh, ww = sim.shape[-2:]
    idx = sim.flatten().argmax().item()
    iy, ix = divmod(idx, ww)
    return (ix + 0.5) / ww * W_t, (iy + 0.5) / hh * H_t


def predict_rows(df, extractor, root):
    preds = []
    for (src, tgt), g in df.groupby(["source_image", "target_image"], sort=False):
        ps, pt = os.path.join(root, src), os.path.join(root, tgt)
        Ws, Hs = Image.open(ps).size
        Wt, Ht = Image.open(pt).size
        fs, ft = extractor(ps), extractor(pt)
        for _, r in g.iterrows():
            f_src = sample_feature(fs, r.source_x, r.source_y, Ws, Hs)
            x, y = match(ft, f_src, Wt, Ht)
            preds.append((r.id, x, y))
    return pd.DataFrame(preds, columns=["id", "x", "y"])


def pck(df, pred, thr=0.10):
    """PCK@thr: distanza al punto accettato piu' vicino / scale < thr. (usa metric.py ufficiale per il valore definitivo)"""
    m = df.merge(pred, on="id")
    ok = []
    for _, r in m.iterrows():
        pts = np.array(json.loads(r.target_points), dtype=float).reshape(-1, 2)
        d = np.linalg.norm(pts - np.array([r.x, r.y]), axis=1).min()
        ok.append(d / r.scale < thr)
    return float(np.mean(ok))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="participant_data")
    ap.add_argument("--mode", choices=["eval", "test"], default="eval")
    ap.add_argument("--n_pairs", type=int, default=60)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    a = ap.parse_args()

    ext = FeatureExtractor(a.model, size=a.size)

    if a.mode == "eval":
        df = pd.read_csv(os.path.join(a.root, "train.csv"))
        pairs = df["pair_id"].drop_duplicates().sample(min(a.n_pairs, df.pair_id.nunique()), random_state=0)
        df = df[df.pair_id.isin(pairs)]
        pred = predict_rows(df, ext, a.root)
        for t in (0.05, 0.10, 0.20):
            print(f"PCK@{t:.2f} = {pck(df, pred, t):.4f}")
    else:
        df = pd.read_csv(os.path.join(a.root, "test.csv"))
        pred = predict_rows(df, ext, a.root)
        pred.to_csv("predictions.csv", index=False)
        print("Salvato predictions.csv (poi: python make_submission.py ...)")