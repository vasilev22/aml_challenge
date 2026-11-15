import json
import numpy as np

def pck_at(pred_xy, target_points, scale, thr=0.10):
    pts = np.array(json.loads(target_points), dtype=float).reshape(-1, 2)
    d = np.linalg.norm(pts - np.array(pred_xy, dtype=float), axis=1).min()
    return d / scale < thr