# -*- coding: utf-8 -*-
"""Shared module for Illusion 3 embedding metrics: silhouette / davies-bouldin / centroid separation ratio (sep) + long-format saving."""
import os
import csv
import numpy as np


def sep_ratio(emb, y):
    """Centroid separation ratio = (mean distance of class centroids to the global centroid) / (mean within-class radius). Larger = more separated."""
    y = np.asarray(y)
    g = emb.mean(0)
    cent, within = [], []
    for c in np.unique(y):
        e = emb[y == c]
        cn = e.mean(0)
        cent.append(cn)
        within.append(float(np.linalg.norm(e - cn, axis=1).mean()))
    between = float(np.linalg.norm(np.array(cent) - g, axis=1).mean())
    return between / (np.mean(within) + 1e-9)


def compute(emb, y, cap=3000, seed=42):
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan"), float("nan"), len(emb), 1
    e, yy = emb, y
    if len(e) > cap:
        idx = np.random.default_rng(seed).permutation(len(e))[:cap]
        e, yy = e[idx], y[idx]
    return (float(silhouette_score(e, yy)), float(davies_bouldin_score(e, yy)),
            sep_ratio(e, yy), len(emb), len(set(y.tolist())))


def save_tsne_coords(path, emb, y, cap=3000, seed=42):
    """Save t-SNE 2D coordinates as (x,y,label) CSV (for presentation figures)."""
    from sklearn.manifold import TSNE
    y = np.asarray(y)
    e, yy = emb, y
    if len(e) > cap:
        idx = np.random.default_rng(seed).permutation(len(e))[:cap]
        e, yy = e[idx], y[idx]
    xy = TSNE(2, init="pca", perplexity=30, random_state=seed).fit_transform(e)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label"])
        for (a, b), lb in zip(xy, yy):
            w.writerow([round(float(a), 3), round(float(b), 3), int(lb)])


def append_long(path, model, ds, exp, split, emb, y):
    """Append 1 long-format row (model,dataset,exp,split,silhouette,davies_bouldin,sep_ratio,n,classes)."""
    sil, db, sep, n, nc = compute(emb, y)
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)   # embeddings/ is not in a fresh clone
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "dataset", "exp", "split",
                        "silhouette", "davies_bouldin", "sep_ratio", "n", "classes"])
        w.writerow([model, ds, exp, split, round(sil, 4), round(db, 3),
                    round(sep, 4), n, nc])
    return sil, sep
