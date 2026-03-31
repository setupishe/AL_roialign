"""
Oracle Distillation Pipeline for Active Object Detection
=========================================================
1. Multiple oracle definitions (GT-based)
2. Oracle disagreement analysis
3. Proxy feature extraction (from embeddings + predictions)
4. GBM training to predict oracle scores from proxy features
5. Evaluation: Spearman rank correlation

Usage:
  python oracle_distillation.py --step 0.05 --to-step 0.075 --embeds-dir <path>
  python oracle_distillation.py --oracle-only  # just compare oracle variants (no embeds needed)
"""
import os, sys, glob, argparse, time, json
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))

DATASETS_DIR = os.environ.get("DATASETS_DIR", "/home/setupishe/datasets")

# =============================================================================
# GT Database
# =============================================================================

def load_gt_labels(labels_dir, img_name):
    lf = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
    if not os.path.exists(lf) or os.path.getsize(lf) == 0:
        return []
    with open(lf) as f:
        lines = [l.strip() for l in f if l.strip()]
    boxes = []
    for line in lines:
        p = line.split()
        boxes.append({
            "cls": int(p[0]),
            "xc": float(p[1]), "yc": float(p[2]),
            "w": float(p[3]), "h": float(p[4]),
            "area": float(p[3]) * float(p[4])
        })
    return boxes


def build_gt_database(dataset_name):
    labels_dir = f"{DATASETS_DIR}/{dataset_name}/labels/train"
    imgs_dir = f"{DATASETS_DIR}/{dataset_name}/images/train"
    db = {}
    for ip in sorted(glob.glob(os.path.join(imgs_dir, "*.jpg"))):
        name = os.path.basename(ip)
        db[name] = load_gt_labels(labels_dir, name)
    return db


def compute_pool_stats(pool_names, gt_db):
    cc = Counter()
    total = 0
    area_sum = 0.0
    for n in pool_names:
        for b in gt_db.get(n, []):
            cc[b["cls"]] += 1
            total += 1
            area_sum += b["area"]
    return {
        "class_counts": cc,
        "total_boxes": total,
        "classes_covered": len(cc),
        "avg_area": area_sum / max(1, total),
        "num_images": len(pool_names),
    }


# =============================================================================
# Oracle Definitions
# =============================================================================

def oracle_a_density(img_name, gt_db, pool_stats):
    """Oracle A: object density + class coverage.
    Score = num_boxes * (1 + num_unique_classes / total_possible_classes)
    Favors images with MANY objects from MANY classes.
    """
    boxes = gt_db.get(img_name, [])
    if not boxes:
        return 0.0
    num_boxes = len(boxes)
    unique_cls = len(set(b["cls"] for b in boxes))
    return num_boxes * (1.0 + unique_cls / 20.0)


def oracle_b_class_balanced(img_name, gt_db, pool_stats):
    """Oracle B: rare class boost.
    Score = sum of (1 / class_frequency_in_pool) for each box.
    Strongly favors images with rare classes.
    """
    boxes = gt_db.get(img_name, [])
    if not boxes:
        return 0.0
    cc = pool_stats["class_counts"]
    total = pool_stats["total_boxes"]
    score = 0.0
    for b in boxes:
        freq = cc.get(b["cls"], 0) / max(1, total)
        score += 1.0 / (freq + 0.001)
    return score


def oracle_c_diversity(img_name, gt_db, pool_stats):
    """Oracle C: class diversity + size diversity.
    Score = unique_classes^2 + size_diversity_bonus
    Favors images that bring new class coverage AND varied object sizes.
    """
    boxes = gt_db.get(img_name, [])
    if not boxes:
        return 0.0
    unique_cls = len(set(b["cls"] for b in boxes))
    avg_pool_area = pool_stats["avg_area"]
    sizes = [b["area"] for b in boxes]
    size_var = float(np.std(sizes)) if len(sizes) > 1 else 0.0
    size_dev = np.mean([abs(s - avg_pool_area) for s in sizes])
    return unique_cls ** 2 + size_var * 10.0 + size_dev * 5.0


def oracle_d_combined(img_name, gt_db, pool_stats):
    """Oracle D: the original v1 oracle (rare + diversity + count + size_div).
    Kept for backward compatibility.
    """
    boxes = gt_db.get(img_name, [])
    if not boxes:
        return 0.0
    cc = pool_stats["class_counts"]
    total = pool_stats["total_boxes"]
    rare = sum(1.0 / (cc.get(b["cls"], 0) / max(1, total) + 0.01) for b in boxes)
    diversity = len(set(b["cls"] for b in boxes))
    count = len(boxes)
    avg_a = pool_stats["avg_area"]
    size_div = sum(abs(b["area"] - avg_a) for b in boxes) / max(1, len(boxes))
    return rare * 1.0 + diversity * 2.0 + count * 0.5 + size_div * 5.0


ORACLES = {
    "A_density": oracle_a_density,
    "B_class_bal": oracle_b_class_balanced,
    "C_diversity": oracle_c_diversity,
    "D_combined": oracle_d_combined,
}


def score_all_images(oracle_fn, unlabeled_imgs, gt_db, pool_stats):
    return {img: oracle_fn(img, gt_db, pool_stats) for img in unlabeled_imgs}


# =============================================================================
# Oracle Disagreement Analysis
# =============================================================================

def oracle_disagreement(oracle_scores_dict, unlabeled_imgs):
    """Compute pairwise Spearman between all oracle variants."""
    names = sorted(oracle_scores_dict.keys())
    n = len(names)
    print(f"\n{'Oracle Disagreement (Spearman ρ)':^60}")
    print("-" * 60)
    header = f"{'':>14}" + "".join(f"{n:>14}" for n in names)
    print(header)
    for i, ni in enumerate(names):
        vals_i = np.array([oracle_scores_dict[ni][img] for img in unlabeled_imgs])
        row = f"{ni:>14}"
        for j, nj in enumerate(names):
            vals_j = np.array([oracle_scores_dict[nj][img] for img in unlabeled_imgs])
            rho, _ = spearmanr(vals_i, vals_j)
            row += f"{rho:>14.4f}"
        print(row)
    print()


def top_k_analysis(oracle_scores_dict, k, gt_db):
    """For each oracle, show what its top-k selection looks like (GT stats)."""
    print(f"\n{'Top-k Selection Analysis (k=' + str(k) + ')':^80}")
    print("-" * 80)
    print(f"{'Oracle':<14} {'BPI':>6} {'Cls':>4} {'Empty':>6} {'TotBox':>7} {'AvgArea':>8} {'ClassEntropy':>13}")
    print("-" * 80)
    for name, scores in sorted(oracle_scores_dict.items()):
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sel = [img for img, _ in ranked[:k]]
        cc = Counter()
        total = 0; areas = []; empty = 0
        for img in sel:
            boxes = gt_db.get(img, [])
            if not boxes:
                empty += 1; continue
            for b in boxes:
                cc[b["cls"]] += 1; total += 1; areas.append(b["area"])
        bpi = total / max(1, len(sel))
        ncls = len(cc)
        avg_area = float(np.mean(areas)) if areas else 0
        # Class entropy (uniform = high)
        if cc:
            probs = np.array(list(cc.values()), dtype=float)
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            entropy = 0
        print(f"  {name:<12} {bpi:>6.2f} {ncls:>4d} {empty:>6d} {total:>7d} {avg_area:>8.4f} {entropy:>13.4f}")


# =============================================================================
# Proxy Feature Extraction (needs embeddings)
# =============================================================================

def extract_proxy_features(embeds_dir, unlabeled_imgs, first_list_set):
    """
    Extract per-image proxy features from embedding files.
    Returns dict: img_stem -> feature_vector
    """
    from al_utils import _slice_vector, build_hnsw_index, get_embedding_dim

    all_npy = sorted(glob.glob(os.path.join(embeds_dir, "*.npy")))
    if not all_npy:
        all_npy = sorted(glob.glob(os.path.join(embeds_dir, "**", "*.npy"), recursive=True))

    first_list, second_list = [], []
    for f in all_npy:
        base = os.path.basename(f)
        img_stem = base[:base.index("_cropped")]
        if img_stem in first_list_set:
            first_list.append(f)
        else:
            second_list.append(f)

    embedding_dim = get_embedding_dim(first_list)
    print(f"Building HNSW index (dim={embedding_dim}, labeled crops={len(first_list)})...")
    index = build_hnsw_index(first_list, embedding_dim)

    # Matryoshka prefix indices
    divs = [8, 4, 2, 1]
    prefix_dims = sorted(set(max(1, embedding_dim // d) for d in divs))

    by_img = defaultdict(list)
    print(f"Scoring {len(second_list)} crops...")
    bs = 1024
    for start in range(0, len(second_list), bs):
        bf = second_list[start:start + bs]
        batch = np.empty((len(bf), embedding_dim), dtype=np.float32)
        for i, f in enumerate(bf):
            batch[i] = _slice_vector(np.load(f).squeeze(0).astype(np.float32), embedding_dim)
        _, dists = index.knn_query(batch, k=1)
        for i, f in enumerate(bf):
            img_stem = os.path.basename(f)[:os.path.basename(f).index("_cropped")]
            vec = batch[i]
            by_img[img_stem].append({
                "dist": float(dists[i][0]),
                "vec": vec,
            })

    # Build per-image feature vectors
    features = {}
    for img_stem, crops in by_img.items():
        dists = [c["dist"] for c in crops]
        vecs = np.array([c["vec"] for c in crops])

        feat = {
            "crop_count": len(crops),
            "dist_max": max(dists),
            "dist_min": min(dists),
            "dist_mean": np.mean(dists),
            "dist_std": np.std(dists) if len(dists) > 1 else 0,
            "dist_sum": sum(dists),
            "dist_median": float(np.median(dists)),
            # Embedding spread within image
            "emb_mean_norm": float(np.linalg.norm(np.mean(vecs, axis=0))),
            "emb_std": float(np.mean(np.std(vecs, axis=0))) if len(vecs) > 1 else 0,
            # Pairwise distance within image (diversity of detections)
            "intra_dist_mean": 0.0,
        }
        if len(vecs) > 1:
            from sklearn.metrics.pairwise import cosine_distances
            pwd = cosine_distances(vecs)
            feat["intra_dist_mean"] = float(np.mean(pwd[np.triu_indices(len(vecs), k=1)]))

        features[img_stem] = feat

    return features


# =============================================================================
# GBM Training
# =============================================================================

def train_gbm_proxy(features, oracle_scores, oracle_name):
    """Train LightGBM to predict oracle score from proxy features."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    imgs = sorted(set(features.keys()) & set(oracle_scores.keys()))
    feat_names = sorted(features[imgs[0]].keys())

    X = np.array([[features[img][fn] for fn in feat_names] for img in imgs])
    y = np.array([oracle_scores[img] for img in imgs])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rhos = []
    importances = np.zeros(len(feat_names))

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feat_names, free_raw_data=False)
        dval = lgb.Dataset(X_val, y_val, feature_name=feat_names, free_raw_data=False)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = lgb.train(
            params, dtrain, num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        pred = model.predict(X_val)
        rho, _ = spearmanr(y_val, pred)
        rhos.append(rho)
        importances += model.feature_importance(importance_type="gain")

    mean_rho = np.mean(rhos)
    importances /= 5

    print(f"\n  GBM → {oracle_name}: Spearman ρ = {mean_rho:.4f} (5-fold CV)")
    print(f"  Feature importances:")
    for fn, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        print(f"    {fn:<20s} {imp:>10.1f}")

    return mean_rho, dict(zip(feat_names, importances))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="VOC")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--to-step", type=float, default=0.075)
    parser.add_argument("--embeds-dir", default=None)
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("ORACLE DISTILLATION PIPELINE")
    print("=" * 70)

    gt_db = build_gt_database(args.dataset)
    total_imgs = len(gt_db)
    print(f"GT database: {total_imgs} images")

    # Load labeled pool
    train_list_path = f"{DATASETS_DIR}/{args.dataset}/train_{args.step}.txt"
    with open(train_list_path) as f:
        pool_names = [os.path.basename(l.strip()) for l in f if l.strip()]
    pool_stems = set(os.path.splitext(n)[0] for n in pool_names)
    pool_stats = compute_pool_stats(pool_names, gt_db)
    print(f"Labeled pool: {len(pool_names)} images, {pool_stats['total_boxes']} boxes, "
          f"{pool_stats['classes_covered']} classes")

    # Unlabeled images
    unlabeled_imgs = sorted([img for img in gt_db.keys() if os.path.splitext(img)[0] not in pool_stems])
    k = int(total_imgs * (args.to_step - args.step))
    print(f"Unlabeled: {len(unlabeled_imgs)} images, will select k={k}")

    # Score all unlabeled images with each oracle
    print("\n--- Oracle Scoring ---")
    oracle_scores = {}
    for name, fn in ORACLES.items():
        t0 = time.time()
        scores = score_all_images(fn, unlabeled_imgs, gt_db, pool_stats)
        oracle_scores[name] = scores
        print(f"  {name}: {time.time()-t0:.1f}s")

    # Oracle disagreement
    oracle_disagreement(oracle_scores, unlabeled_imgs)

    # Top-k analysis
    top_k_analysis(oracle_scores, k, gt_db)

    if args.oracle_only:
        print("\n--oracle-only mode: stopping here.")
        return

    # Proxy features (need embeddings)
    if args.embeds_dir is None or not os.path.isdir(args.embeds_dir):
        print(f"\nNo embeddings at {args.embeds_dir}. Cannot extract proxy features or train GBM.")
        print("Re-run with --embeds-dir pointing to extracted embeddings.")
        return

    print("\n--- Proxy Feature Extraction ---")
    features = extract_proxy_features(args.embeds_dir, unlabeled_imgs, pool_stems)
    print(f"Extracted features for {len(features)} images")

    # Also compare raw proxy features vs oracles (without GBM)
    print("\n--- Raw Proxy Feature vs Oracle Correlations ---")
    feat_names = sorted(next(iter(features.values())).keys())
    imgs = sorted(set(features.keys()) & set(unlabeled_imgs))
    print(f"{'Feature':<22}", end="")
    for oname in sorted(oracle_scores.keys()):
        print(f"{oname:>14}", end="")
    print()
    for fn in feat_names:
        fvals = np.array([features.get(os.path.splitext(img)[0], {}).get(fn, 0) for img in imgs])
        print(f"  {fn:<20}", end="")
        for oname in sorted(oracle_scores.keys()):
            ovals = np.array([oracle_scores[oname][img] for img in imgs])
            rho, _ = spearmanr(fvals, ovals)
            print(f"{rho:>14.4f}", end="")
        print()

    # Train GBM for each oracle
    print("\n--- GBM Proxy Training ---")
    # Convert features keys to match oracle keys (add .jpg)
    features_jpg = {}
    for stem, feat in features.items():
        features_jpg[stem + ".jpg"] = feat

    for oname in sorted(oracle_scores.keys()):
        train_gbm_proxy(features_jpg, oracle_scores[oname], oname)


if __name__ == "__main__":
    main()
