"""GBM proxy — uses pre-concatenated embeddings."""
import os, sys, glob, time
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import spearmanr

DATASETS_DIR = "/home/setupishe/datasets"
DATASET = "VOC"
LABELS_DIR = f"{DATASETS_DIR}/{DATASET}/labels/train"
CONCAT_FILE = "/tmp/embeds_concat.npz"
STEP_FROM = 0.1
STEP_TO = 0.125
USE_DIMS = [448, 896, 1792, 3584]  # dim//32, dim//16, dim//8, dim//4

def load_gt(img_name):
    lf = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))
    if not os.path.exists(lf) or os.path.getsize(lf) == 0: return []
    boxes = []
    with open(lf) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: boxes.append({"cls": int(p[0]), "area": float(p[3])*float(p[4])})
    return boxes

def build_gt_db():
    db = {}
    for ip in sorted(glob.glob(f"{DATASETS_DIR}/{DATASET}/images/train/*.jpg")):
        db[os.path.basename(ip)] = load_gt(os.path.basename(ip))
    return db

def oracle_density(boxes, cc, tot):
    if not boxes: return 0.0
    return len(boxes) * (1.0 + len(set(b["cls"] for b in boxes)) / 20.0)

def oracle_class_bal(boxes, cc, tot):
    if not boxes: return 0.0
    return sum(1.0 / (cc.get(b["cls"], 0) / max(1, tot) + 0.001) for b in boxes)

def oracle_balanced(boxes, cc, tot):
    if not boxes: return 0.0
    n = len(boxes)
    rarity = sum(1.0 / (cc.get(b["cls"], 0) / max(1, tot) + 0.01) for b in boxes) / n
    unique = len(set(b["cls"] for b in boxes))
    return n * 0.5 + rarity * 0.3 + unique * 2.0

ORACLES = {"density": oracle_density, "class_bal": oracle_class_bal, "balanced": oracle_balanced}

def l2_norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)

def main():
    t0 = time.time()
    print("=" * 70)
    print("GBM PROXY — MATRYOSHKA GRANULARITY")
    print("=" * 70)

    gt_db = build_gt_db()
    with open(f"{DATASETS_DIR}/{DATASET}/train_{STEP_FROM}.txt") as f:
        pool_names = [os.path.basename(l.strip()) for l in f if l.strip()]
    pool_stems = set(os.path.splitext(n)[0] for n in pool_names)
    cc = Counter(); tot = 0
    for n in pool_names:
        for b in gt_db.get(n, []): cc[b["cls"]] += 1; tot += 1
    unlabeled = sorted([img for img in gt_db if os.path.splitext(img)[0] not in pool_stems])
    k = int(len(gt_db) * (STEP_TO - STEP_FROM))
    print(f"Pool: {len(pool_names)}, Unlabeled: {len(unlabeled)}, k={k}")

    oracle_scores = {}
    for oname, ofn in ORACLES.items():
        oracle_scores[oname] = {img: ofn(gt_db.get(img, []), cc, tot) for img in unlabeled}

    # Load pre-concatenated embeddings
    print(f"\nLoading {CONCAT_FILE}...")
    data = np.load(CONCAT_FILE)
    all_vecs = data["vecs"]
    stems = data["stems"].tolist()
    print(f"  Shape: {all_vecs.shape}, {all_vecs.nbytes/1e6:.0f} MB")

    labeled_mask = np.array([s in pool_stems for s in stems])
    L_vecs = all_vecs[labeled_mask]
    U_vecs = all_vecs[~labeled_mask]
    U_stems = [s for s, m in zip(stems, ~labeled_mask) if m]
    print(f"  Labeled: {L_vecs.shape[0]}, Unlabeled: {U_vecs.shape[0]}")

    # 1-NN at each matryoshka prefix
    print(f"\n1-NN at dims {USE_DIMS}...")
    nn_dists = {}
    for d in USE_DIMS:
        t1 = time.time()
        L = l2_norm(L_vecs[:, :d])
        U = l2_norm(U_vecs[:, :d])
        dists = np.empty(U.shape[0], dtype=np.float32)
        for s in range(0, U.shape[0], 1000):
            e = min(s + 1000, U.shape[0])
            sim = U[s:e] @ L.T
            dists[s:e] = 1.0 - np.max(sim, axis=1)
        nn_dists[d] = dists
        print(f"  dim={d}: {time.time()-t1:.1f}s, mean={np.mean(dists):.4f}")

    # Per-image features
    print(f"\nAggregating per-image...")
    by_img = defaultdict(list)
    for i, stem in enumerate(U_stems):
        by_img[stem].append(i)

    features = {}
    for stem, idxs in by_img.items():
        feat = {"crop_count": len(idxs)}
        for d in USE_DIMS:
            ds = nn_dists[d][idxs]
            feat[f"d{d}_max"] = float(np.max(ds))
            feat[f"d{d}_mean"] = float(np.mean(ds))
            feat[f"d{d}_sum"] = float(np.sum(ds))
            feat[f"d{d}_std"] = float(np.std(ds)) if len(ds) > 1 else 0.0
        # Matryoshka cross-scale features
        d_lo = nn_dists[USE_DIMS[0]][idxs]
        d_hi = nn_dists[USE_DIMS[-1]][idxs]
        diffs = d_hi - d_lo
        feat["matr_diff_mean"] = float(np.mean(diffs))
        feat["matr_diff_std"] = float(np.std(diffs)) if len(diffs) > 1 else 0.0
        per_crop_var = np.var(np.stack([nn_dists[d][idxs] for d in USE_DIMS], axis=0), axis=0)
        feat["matr_var_mean"] = float(np.mean(per_crop_var))
        feat["matr_var_max"] = float(np.max(per_crop_var))
        feat["matr_var_sum"] = float(np.sum(per_crop_var))
        if len(idxs) > 2:
            rho, _ = spearmanr(d_lo, d_hi)
            feat["scale_consistency"] = float(rho) if not np.isnan(rho) else 0.0
        else:
            feat["scale_consistency"] = 0.0
        features[stem] = feat

    feat_names = sorted(next(iter(features.values())).keys())
    print(f"  {len(features)} images, {len(feat_names)} features")

    # Raw correlations
    imgs = sorted(set(features.keys()) & set(os.path.splitext(img)[0] for img in unlabeled))
    print(f"\n{'Feature → Oracle (Spearman)':^80}")
    print("-" * 80)
    hdr = f"{'Feature':<26}"
    for o in sorted(oracle_scores): hdr += f"{o:>18}"
    print(hdr)
    for fn in feat_names:
        fv = np.array([features[img][fn] for img in imgs])
        row = f"  {fn:<24}"
        for o in sorted(oracle_scores):
            ov = np.array([oracle_scores[o].get(img + ".jpg", 0) for img in imgs])
            rho, _ = spearmanr(fv, ov)
            row += f"{rho:>18.4f}"
        print(row)

    # GBM
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    print(f"\n{'GBM 5-fold CV':^60}")
    print("=" * 60)
    X = np.array([[features[img][fn] for fn in feat_names] for img in imgs])
    for oname in sorted(oracle_scores):
        y = np.array([oracle_scores[oname].get(img + ".jpg", 0) for img in imgs])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rhos = []; imps = np.zeros(len(feat_names))
        for tri, vai in kf.split(X):
            dt = lgb.Dataset(X[tri], y[tri], feature_name=feat_names, free_raw_data=False)
            dv = lgb.Dataset(X[vai], y[vai], feature_name=feat_names, free_raw_data=False)
            m = lgb.train({"objective": "regression", "metric": "mae", "lr": 0.05,
                 "num_leaves": 31, "verbose": -1, "n_jobs": -1},
                dt, 300, valid_sets=[dv], callbacks=[lgb.early_stopping(20, verbose=False)])
            pred = m.predict(X[vai])
            rho, _ = spearmanr(y[vai], pred)
            rhos.append(rho); imps += m.feature_importance(importance_type="gain")
        imps /= 5
        print(f"\n  {oname}: ρ = {np.mean(rhos):.4f} (folds: {[f'{r:.3f}' for r in rhos]})")
        for fn, v in sorted(zip(feat_names, imps), key=lambda x: -x[1])[:5]:
            print(f"    {fn:<26} {v:>10.1f}")

    print(f"\nDone in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
