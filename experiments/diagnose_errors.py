"""
Diagnosi degli errori della pipeline (nessuna GPU, secondi). Va messo in experiments/ e lanciato dalla
radice del repo, come run_best_pipeline.py.

    python experiments/diagnose_errors.py --cache /kaggle/input/datasets/axelromanelli/cache-advml/cache

--cache : cartella di sola lettura con le mappe ViT-B (zoom) -> dataset Kaggle
--work  : cartella dove sono state scritte testa e refine (default "cache")
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_maps import load_maps, blend, predict, target_array, errors  # noqa: E402

C = "s0.25_t0.15_m0.25_rect"
BINS = [0, 0.05, 0.10, 0.15, 0.30, np.inf]
LABELS = ["<0.05", "0.05-0.10", "0.10-0.15", "0.15-0.30", ">0.30"]


def run(paths, ids, weights, alpha, method):
    z, g, r = load_maps(paths, ids)
    return predict(blend(z, g, weights, alpha), r, method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--work", default="cache")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="experiments/results/diagnosis_per_query.csv")
    args = ap.parse_args()

    q = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    ids = q["id"].tolist()
    targets, scale = target_array(q), q["scale"].to_numpy()

    vitb = f"{args.cache}/simmaps/train_dinov2_vitb14_798_fp16_{C}.pt"
    head = f"{args.work}/simmaps/train_head_conv_h384_o256_nested.pt"
    refine = [f"{args.work}/simmaps/train_refine_{b}_448_f0.4_coarse_BHF.pt"
              for b in ("dinov2_vitb14", "dinov2_vitl14_hf")]

    W5, W3 = ("window", 2, 0.05), ("window", 2, 0.03)
    preds = {
        "vitb_raw": run([vitb], ids, [1.0], 0.0, W5),
        "head": run([head], ids, [1.0], 0.0, W5),
        "coarse(.25B+.75H)": run([vitb, head], ids, [0.25, 0.75], 0.0, W3),
        "final(refine)": run(refine, ids, [0.75, 0.25], 0.6, W3),
    }
    errs = {k: errors(p, targets, scale) for k, p in preds.items()}

    print("=== PCK@0.10 / PCK@0.05 per stadio ===")
    for k, e in errs.items():
        print(f"{k:20s} {np.mean(e < .10):.4f}   {np.mean(e < .05):.4f}")

    print("\n=== Istogramma errore (frazione delle query) ===")
    hist = pd.DataFrame({k: pd.Series(pd.cut(e, BINS, labels=LABELS, right=False)).value_counts(normalize=True)
                         for k, e in errs.items()}).reindex(LABELS)
    print(hist.round(3).to_string())

    e, pred = errs["final(refine)"], preds["final(refine)"]
    wrong = e >= 0.10
    print(f"\nFinale: {wrong.sum()} errori su {len(e)}  "
          f"(quasi giusti 0.10-0.15: {np.sum((e >= .10) & (e < .15))}, "
          f"sbagliati >0.30: {np.sum(e >= .30)}, errore mediano tra gli sbagliati >0.30: "
          f"{np.median(e[e >= .30]) if np.any(e >= .30) else float('nan'):.2f})")

    # scambio con un altro keypoint della stessa coppia (tipico di simmetrie destra/sinistra)
    pair = q["pair_id"].to_numpy()
    swap = np.zeros(len(q), bool)
    for p in np.unique(pair):
        idx = np.where(pair == p)[0]
        for i in idx:
            if not wrong[i] or len(idx) < 2:
                continue
            others = targets[[j for j in idx if j != i]].reshape(-1, 2)
            others = others[~np.isnan(others).any(1)]
            if len(others):
                swap[i] = np.sqrt(((others - pred[i]) ** 2).sum(1)).min() / scale[i] < 0.10
    print(f"Errori in cui la predizione cade su un ALTRO keypoint della stessa coppia: "
          f"{swap.sum()} ({swap.sum() / max(wrong.sum(), 1):.0%} degli errori)")

    # per dimensione relativa (stesse fasce di evaluate_maps)
    sizes = {p["pair_id"]: max(p["target_size"]) for p in
             json.load(open(os.path.join(args.data_dir, "train_pairs.json")))["pairs"]}
    rel = scale / q["pair_id"].map(sizes).to_numpy()
    bins = pd.cut(rel, [0, .2, .4, .7, 10])
    print("\n=== PCK finale per dimensione relativa (scale / lato target) ===")
    print(pd.DataFrame({"pck": pd.Series(~wrong).groupby(bins, observed=True).mean(),
                        "n": pd.Series(~wrong).groupby(bins, observed=True).size()}).round(3).to_string())

    # chi aiuta chi: coarse -> final, e oracolo
    a, b = errs["coarse(.25B+.75H)"] < .10, errs["final(refine)"] < .10
    print(f"\n=== Raffinamento: coarse -> final ===\n"
          f"giuste->sbagliate: {np.sum(a & ~b)}   sbagliate->giuste: {np.sum(~a & b)}   "
          f"(netto {np.sum(~a & b) - np.sum(a & ~b):+d})")
    orac = np.any([e_ < .10 for e_ in errs.values()], axis=0)
    print(f"Oracolo (giusta in almeno uno stadio): {orac.mean():.4f}  "
          f"-> margine massimo di un'eventuale selezione/ensemble")

    # coppie peggiori
    g = pd.DataFrame({"pair_id": pair, "ok": ~wrong}).groupby("pair_id")["ok"].agg(["mean", "size"])
    print("\n=== 10 coppie peggiori (>=3 query) ===")
    print(g[g["size"] >= 3].sort_values("mean").head(10).round(2).to_string())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame({"id": ids, "pair_id": pair, "scale": scale, "rel_size": rel,
                       "x": pred[:, 0], "y": pred[:, 1], "swap_other_kp": swap})
    for k, v in errs.items():
        df[f"err_{k}"] = v
    df.to_csv(args.out, index=False)
    print(f"\nSalvato {args.out} (errori per query, per ispezione manuale)")


if __name__ == "__main__":
    main()
