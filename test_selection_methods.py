"""
Test multiple selection methods against oracle using RANK CORRELATION.
Instead of set overlap, we score ALL unlabeled images by each method
and by oracle, then compute Spearman rho between rankings.
"""
import os, glob, sys, time
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

sys.path.insert(0, "/home/setupishe/bel_conf")
from al_utils import (
    _slice_vector, _load_matrix, build_hnsw_index,
    get_embedding_dim,
)

DATASETS_DIR = "/home/setupishe/datasets"
DATASET = "VOC"
NUM_CLASSES = 20
LABELS_DIR = f"{DATASETS_DIR}/{DATASET}/labels/train"
EMBEDS_DIR = f"{DATASETS_DIR}/reduced_embeds_0.05_experiment_matr"
FROM_FRACTION = 0.05
TO_FRACTION = 0.075
TRAIN_LIST = f"{DATASETS_DIR}/{DATASET}/train_{FROM_FRACTION}.txt"


def load_train_list(path):
    with open(path) as f:
        return [os.path.basename(l.strip()) for l in f if l.strip()]


def load_gt_labels(img_name):
    lf = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))
    if not os.path.exists(lf) or os.path.getsize(lf) == 0:
        return []
    with open(lf) as f:
        lines = [l.strip() for l in f if l.strip()]
    boxes = []
    for line in lines:
        p = line.split()
        boxes.append({"cls": int(p[0]), "xc": float(p[1]), "yc": float(p[2]),
                       "w": float(p[3]), "h": float(p[4]), "area": float(p[3])*float(p[4])})
    return boxes


def build_gt_database():
    db = {}
    for ip in tqdm(sorted(glob.glob(f"{DATASETS_DIR}/{DATASET}/images/train/*.jpg")), desc="GT"):
        db[os.path.basename(ip)] = load_gt_labels(os.path.basename(ip))
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
    return {"class_counts": cc, "total_boxes": total, "avg_area": area_sum / max(1, total)}


def oracle_score_image(img_name, gt_db, pool_stats):
    """Score how much info this image adds (using GT). Higher = better."""
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


def build_file_lists(embeds_dir, from_names_set):
    all_npy = sorted(glob.glob(os.path.join(embeds_dir, "*.npy")))
    if not all_npy:
        all_npy = sorted(glob.glob(os.path.join(embeds_dir, "**", "*.npy"), recursive=True))
    first_list, second_list = [], []
    for f in all_npy:
        base = os.path.basename(f)
        img_name = base[:base.index("_cropped")]
        if img_name in from_names_set:
            first_list.append(f)
        else:
            second_list.append(f)
    return first_list, second_list


# ==========================================
# SCORING FUNCTIONS (return dict: img_name -> score for ALL unlabeled images)
# ==========================================

def score_distance_max(first_list, second_list, embedding_dim):
    """Standard: image score = max crop distance (first-crop-wins equivalent)."""
    index = build_hnsw_index(first_list, embedding_dim)
    by_img = defaultdict(list)
    bs = 1024
    for start in range(0, len(second_list), bs):
        bf = second_list[start:start+bs]
        batch = np.empty((len(bf), embedding_dim), dtype=np.float32)
        for i, f in enumerate(bf):
            batch[i] = _slice_vector(np.load(f).squeeze(0).astype(np.float32), embedding_dim)
        _, dists = index.knn_query(batch, k=1)
        for i, f in enumerate(bf):
            img = os.path.basename(f)[:os.path.basename(f).index("_cropped")]
            by_img[img].append(float(dists[i][0]))
    return {img: max(ds) for img, ds in by_img.items()}


def score_distance_mean(first_list, second_list, embedding_dim):
    """Image score = mean crop distance."""
    index = build_hnsw_index(first_list, embedding_dim)
    by_img = defaultdict(list)
    bs = 1024
    for start in range(0, len(second_list), bs):
        bf = second_list[start:start+bs]
        batch = np.empty((len(bf), embedding_dim), dtype=np.float32)
        for i, f in enumerate(bf):
            batch[i] = _slice_vector(np.load(f).squeeze(0).astype(np.float32), embedding_dim)
        _, dists = index.knn_query(batch, k=1)
        for i, f in enumerate(bf):
            img = os.path.basename(f)[:os.path.basename(f).index("_cropped")]
            by_img[img].append(float(dists[i][0]))
    return {img: np.mean(ds) for img, ds in by_img.items()}


def score_distance_sum(first_list, second_list, embedding_dim):
    """Image score = sum of crop distances (naturally favors more objects)."""
    index = build_hnsw_index(first_list, embedding_dim)
    by_img = defaultdict(list)
    bs = 1024
    for start in range(0, len(second_list), bs):
        bf = second_list[start:start+bs]
        batch = np.empty((len(bf), embedding_dim), dtype=np.float32)
        for i, f in enumerate(bf):
            batch[i] = _slice_vector(np.load(f).squeeze(0).astype(np.float32), embedding_dim)
        _, dists = index.knn_query(batch, k=1)
        for i, f in enumerate(bf):
            img = os.path.basename(f)[:os.path.basename(f).index("_cropped")]
            by_img[img].append(float(dists[i][0]))
    return {img: sum(ds) for img, ds in by_img.items()}


def score_distance_crop_weighted(first_list, second_list, embedding_dim):
    """mean_distance * sqrt(num_crops)"""
    index = build_hnsw_index(first_list, embedding_dim)
    by_img = defaultdict(list)
    bs = 1024
    for start in range(0, len(second_list), bs):
        bf = second_list[start:start+bs]
        batch = np.empty((len(bf), embedding_dim), dtype=np.float32)
        for i, f in enumerate(bf):
            batch[i] = _slice_vector(np.load(f).squeeze(0).astype(np.float32), embedding_dim)
        _, dists = index.knn_query(batch, k=1)
        for i, f in enumerate(bf):
            img = os.path.basename(f)[:os.path.basename(f).index("_cropped")]
            by_img[img].append(float(dists[i][0]))
    return {img: np.mean(ds) * np.sqrt(len(ds)) for img, ds in by_img.items()}


def score_matryoshka_variance(first_list, second_list, embedding_dim):
    """Matryoshka variance: variance of 1-NN dist across prefix widths."""
    divs = [8, 4, 2, 1]
    dims = sorted(set(max(1, embedding_dim // d) for d in divs))
    mats = {d: _load_matrix(first_list, use_dim=d, normalize=True) for d in dims}
    by_img = defaultdict(list)
    for f in tqdm(second_list, desc="matr_var", leave=False):
        vec = np.load(f).squeeze(0).astype(np.float32)
        per_scale = [float(np.min(1.0 - mats[d] @ _slice_vector(vec, d))) for d in dims]
        var = float(np.var(per_scale))
        img = os.path.basename(f)[:os.path.basename(f).index("_cropped")]
        by_img[img].append(var)
    return {img: np.mean(vs) for img, vs in by_img.items()}


def score_prefix_distance(first_list, second_list, embedding_dim, div=2):
    """Distance using shorter matryoshka prefix."""
    use_dim = max(1, embedding_dim // div)
    index = build_hnsw_index(first_list, embedding_dim, use_dim=use_dim)
    by_img = defaultdict(list)
    bs = 1024
    for start in range(0, len(second_list), bs):
        bf = second_list[start:start+bs]
        batch = np.empty((len(bf), use_dim), dtype=np.float32)
        for i, f in enumerate(bf):
            batch[i] = _slice_vector(np.load(f).squeeze(0).astype(np.float32), use_dim)
        _, dists = index.knn_query(batch, k=1)
        for i, f in enumerate(bf):
            img = os.path.basename(f)[:os.path.basename(f).index("_cropped")]
            by_img[img].append(float(dists[i][0]))
    return {img: np.mean(ds) for img, ds in by_img.items()}


def combine_scores_hybrid(dist_scores, var_scores, crop_counts, alpha=1.0, crop_weight=0.5):
    """Percentile-rank product of distance, variance, and crop count."""
    imgs = sorted(dist_scores.keys())
    d = np.array([dist_scores[i] for i in imgs])
    v = np.array([var_scores.get(i, 0) for i in imgs])
    c = np.array([crop_counts.get(i, 1) for i in imgs])
    n = len(imgs)
    d_pct = np.argsort(np.argsort(d)) / max(1, n - 1)
    v_pct = np.argsort(np.argsort(v)) / max(1, n - 1)
    c_pct = np.argsort(np.argsort(c)) / max(1, n - 1)
    combined = d_pct * (v_pct ** alpha) * (c_pct ** crop_weight)
    return {imgs[i]: float(combined[i]) for i in range(n)}


def compute_rank_correlation(scores_a, scores_b):
    """Spearman rank correlation between two score dicts (same keys)."""
    imgs = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    a = np.array([scores_a[i] for i in imgs])
    b = np.array([scores_b[i] for i in imgs])
    rho, p = spearmanr(a, b)
    return rho


def top_k_overlap(scores_a, scores_b, k):
    """Fraction of top-k from A that appear in top-k of B."""
    ranked_a = sorted(scores_a.items(), key=lambda x: x[1], reverse=True)
    ranked_b = sorted(scores_b.items(), key=lambda x: x[1], reverse=True)
    top_a = set(x[0] for x in ranked_a[:k])
    top_b = set(x[0] for x in ranked_b[:k])
    return len(top_a & top_b) / max(1, k)


def analyze_top_k(scores, k, gt_db):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    sel = [img + ".jpg" for img, _ in ranked[:k]]
    cc = Counter()
    total = 0; areas = []; empty = 0
    for name in sel:
        boxes = gt_db.get(name, [])
        if not boxes: empty += 1; continue
        for b in boxes: cc[b["cls"]] += 1; total += 1; areas.append(b["area"])
    return {
        "bpi": total / max(1, len(sel)),
        "classes": len(cc),
        "empty": empty,
        "total_boxes": total,
        "avg_area": float(np.mean(areas)) if areas else 0,
    }


def main():
    print("=" * 70)
    print("SELECTION METHOD EVALUATION — RANK CORRELATION")
    print("=" * 70)

    gt_db = build_gt_database()
    all_names = sorted(gt_db.keys())
    from_names = load_train_list(TRAIN_LIST)
    from_names_set = set(os.path.splitext(n)[0] for n in from_names)
    total_imgs = len(glob.glob(f"{DATASETS_DIR}/{DATASET}/images/train/*.jpg"))
    k = int(total_imgs * (TO_FRACTION - FROM_FRACTION))
    print(f"Pool: {len(from_names)}, Select: {k}")

    # Build file lists
    first_list, second_list = build_file_lists(EMBEDS_DIR, from_names_set)
    print(f"Labeled crops: {len(first_list)}, Unlabeled crops: {len(second_list)}")

    # Crop counts per image
    crop_counts = Counter()
    for f in second_list:
        img = os.path.basename(f)[:os.path.basename(f).index("_cropped")]
        crop_counts[img] += 1
    unlabeled_imgs = sorted(crop_counts.keys())
    print(f"Unlabeled images: {len(unlabeled_imgs)}")
    print(f"Crops/img: mean={np.mean(list(crop_counts.values())):.1f}, "
          f"med={np.median(list(crop_counts.values())):.0f}, max={max(crop_counts.values())}")

    # Oracle scores for all unlabeled images
    print("\nComputing oracle scores...")
    pool_stats = compute_pool_stats(from_names, gt_db)
    oracle_scores = {}
    for img in tqdm(unlabeled_imgs, desc="Oracle"):
        oracle_scores[img] = oracle_score_image(img + ".jpg", gt_db, pool_stats)

    embedding_dim = get_embedding_dim(first_list)
    print(f"Embedding dim: {embedding_dim}")

    # Compute all method scores
    results = {}

    print("\n[1/8] distance_max (standard C2F behavior)...")
    t0 = time.time()
    s = score_distance_max(first_list, second_list, embedding_dim)
    results["dist_max"] = s
    print(f"  {time.time()-t0:.1f}s")

    print("[2/8] distance_mean...")
    t0 = time.time()
    s = score_distance_mean(first_list, second_list, embedding_dim)
    results["dist_mean"] = s
    print(f"  {time.time()-t0:.1f}s")

    print("[3/8] distance_sum...")
    t0 = time.time()
    s = score_distance_sum(first_list, second_list, embedding_dim)
    results["dist_sum"] = s
    print(f"  {time.time()-t0:.1f}s")

    print("[4/8] distance_crop_weighted (mean * sqrt(crops))...")
    t0 = time.time()
    s = score_distance_crop_weighted(first_list, second_list, embedding_dim)
    results["dist_crop_wt"] = s
    print(f"  {time.time()-t0:.1f}s")

    print("[5/8] matryoshka_variance...")
    t0 = time.time()
    s = score_matryoshka_variance(first_list, second_list, embedding_dim)
    results["matr_var"] = s
    print(f"  {time.time()-t0:.1f}s")

    print("[6/8] prefix_div2 (half-dim distance)...")
    t0 = time.time()
    s = score_prefix_distance(first_list, second_list, embedding_dim, div=2)
    results["prefix_d2"] = s
    print(f"  {time.time()-t0:.1f}s")

    print("[7/8] prefix_div4 (quarter-dim distance)...")
    t0 = time.time()
    s = score_prefix_distance(first_list, second_list, embedding_dim, div=4)
    results["prefix_d4"] = s
    print(f"  {time.time()-t0:.1f}s")

    # Hybrids
    print("[8/8] hybrids...")
    for alpha in [0.3, 0.5, 1.0, 2.0]:
        for cw in [0.3, 0.5, 1.0]:
            name = f"hybrid_a{alpha}_c{cw}"
            results[name] = combine_scores_hybrid(
                results["dist_mean"], results["matr_var"],
                dict(crop_counts), alpha=alpha, crop_weight=cw)

    # Also try: just crop count as score (pure object-count selection)
    results["crop_count_only"] = {img: float(crop_counts.get(img, 0)) for img in unlabeled_imgs}

    # Rank correlations
    print("\n" + "=" * 100)
    print(f"{'Method':<28} {'Spearman':>9} {'Top-k overlap':>14} {'BPI':>6} {'Cls':>4} {'Empty':>6} {'TotBox':>7} {'AvgArea':>8}")
    print("-" * 100)

    rows = []
    for name, scores in sorted(results.items()):
        rho = compute_rank_correlation(oracle_scores, scores)
        topk = top_k_overlap(oracle_scores, scores, k)
        q = analyze_top_k(scores, k, gt_db)
        rows.append((rho, name, topk, q))

    rows.sort(key=lambda x: x[0], reverse=True)
    for rho, name, topk, q in rows:
        print(f"  {name:<26} {rho:>9.4f} {topk:>14.4f} {q['bpi']:>6.2f} {q['classes']:>4d} {q['empty']:>6d} {q['total_boxes']:>7d} {q['avg_area']:>8.4f}")

    # Oracle reference
    oq = analyze_top_k(oracle_scores, k, gt_db)
    print(f"  {'ORACLE':<26} {'1.0000':>9} {'1.0000':>14} {oq['bpi']:>6.2f} {oq['classes']:>4d} {oq['empty']:>6d} {oq['total_boxes']:>7d} {oq['avg_area']:>8.4f}")


if __name__ == "__main__":
    main()
